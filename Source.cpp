#include <iostream>
#include <cstdlib>
#include <thread>
#include <string>
#include <filesystem>
#include <iomanip>
#include <vector>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>


/*
	block_width
	___|___
	|     |
	# # # # # # # # --
	# # # # # # # #	 |
	# # # # # # # #	 |_____block_height
	# # # # # # # # --
	# # # # # # # #
	# # # # # # # #
	# # # # # # # #
	# # # # # # # #

*/

class AutoencoderModel
{
private:
	const int image_type = CV_64FC3;
	const int weight_type = CV_64FC1;

	// Weight matrices
	cv::Mat weight1;
	cv::Mat weight2;

	// Block parameters
	int block_width;
	int block_height;
	int block_area;
	int block_volume;

	// Image parameters
	int image_width;
	int image_height;
	int image_channels;

	// train data
	std::vector<cv::Mat> images;


public:
	AutoencoderModel(int image_width, int image_height, int image_channels, int block_width, int block_height) 
		: image_width(image_width), image_height(image_height), image_channels(image_channels),
		block_width(block_width), block_height(block_height), block_area(block_width * block_height),
		block_volume(block_area * image_channels)
	{
		weight1 = cv::Mat(block_volume / 2, block_volume, weight_type);
		cv::randu(weight1, 0, 1);

		weight2 = cv::Mat(block_volume, block_volume / 2, weight_type);
		cv::randu(weight2, 0, 1);
	}

	void sampleData(std::string path="images")
	{
		std::filesystem::path images_directory{ path };

		for (auto& file : std::filesystem::directory_iterator{ images_directory })
		{
			std::string filename = file.path().u8string();

			cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

			image.convertTo(image, image_type);

			if (image.empty())
			{
				std::cerr << "error: can't open image.";
				exit(1);
			}

			if (image.size().width != image_width || image.size().height != image_height)
			{
				remove(filename.c_str());
				std::cerr << "Image dimension is wrong: " << filename << " | File will be deleted" << "\n";
				continue;
			}

			image.convertTo(image, image_type);
			image = image * 2 / 255.0 - 1; // Regularize image

			images.push_back(image);
		}
	}
	
	void sampleData(const char* path) { sampleData(std::string(path)); }

	void train(int epochs=1)
	{
		for (int epoch = 0; epoch < epochs; epoch++)
		{
			std::cout << "Epoch " << epoch + 1 << " =---------------------------------------=\n";
			for (int k = 0; k < images.size(); k++)
			{
				std::chrono::time_point start = std::chrono::system_clock::now(); // Start time of learning

				cv::Mat& image = images[k];

				std::vector<double> errors;

				// Dividing image by blocks
				for (int i = 0; i < image_height / block_height; i++)
					for (int j = 0; j < image_width / block_width; j++)
					{

						cv::Mat block = cv::Mat(image,
							cv::Rect(j * block_width,
								i * block_height,
								block_width,
								block_height));

						cv::Mat vector = cv::Mat(block_area * image_channels, 1, weight_type);

						for (int row = 0; row < block_height; row++)
							for (int col = 0; col < block_width; col++)
								for (int dim = 0; dim < image_channels; dim++)
									vector.at<double>(dim + image_channels * (row + col)) = block.at<cv::Vec3d>(row, col)[dim];
						//vector.at<double>(dim + image_channels * (row + block_width * col)) = block.at<double>(row, col);

						//std::cout << vector << "\n";
						//exit(1);

						cv::Mat values_of_layer1 = weight1 * vector;
						cv::Mat output_values = weight2 * values_of_layer1;

						cv::Mat delta = output_values - vector;

						cv::Mat tmp = delta.t() * delta;
						double error = tmp.at<double>(0, 0);

						errors.push_back(error);

						tmp = 1 / (values_of_layer1.t() * values_of_layer1);
						double learning_rate2 = tmp.at<double>(0, 0);

						tmp = 1 / (vector.t() * vector);
						double learning_rate1 = tmp.at<double>(0, 0);

						weight2 = weight2 - learning_rate2 * delta * values_of_layer1.t();
						weight1 = weight1 - learning_rate1 * weight2.t() * delta * vector.t();

						for (int i = 0; i < weight2.rows; i++)
						{
							cv::Mat length_tmp = weight2.row(i) * weight2.row(i).t();
							double length = length_tmp.at<double>(0, 0);

							weight2.row(i) /= length;
						}

						for (int i = 0; i < weight1.rows; i++)
						{
							cv::Mat length_tmp = weight1.row(i) * weight1.row(i).t();
							double length = length_tmp.at<double>(0, 0);

							weight1.row(i) /= length;
						}
					}

				double sum_error = 0.0;
				for (int i = 0; i < errors.size(); i++)
					sum_error += errors[i];

				long seconds = ((std::chrono::system_clock::now() - start).count() / 1000000)
					* (images.size() * epochs - epoch * images.size() - k - 1);
				long minutes = seconds / 60;
				seconds -= minutes * 60;

				std::cout << "Processing image(" << k + 1 << "/" << images.size() << "): " << std::setw(40) << std::to_string(sum_error)
					<< " time left: " << minutes << "m " << seconds << "s" << "\n";
			}
		}
	}

	void forwardPropogation(std::string resultPath="output")
	{
		for (int k = 0; k < images.size(); k++)
		{
			cv::Mat image = images[k];

			cv::Mat recreated_image = cv::Mat(image_width, image_height, image_type); // Output result of neural network
			for (int i = 0; i < image_height / block_height; i++)
				for (int j = 0; j < image_width / block_width; j++)
				{
					cv::Mat block = cv::Mat(image,
						cv::Rect(j * block_width,
							i * block_height,
							block_width,
							block_height));

					cv::Mat vector = cv::Mat(block_volume, 1, weight_type);

					for (int row = 0; row < block_height; row++)
						for (int col = 0; col < block_width; col++)
							for (int dim = 0; dim < image_channels; dim++)
								vector.at<double>(dim + image_channels * (row + col)) = block.at<cv::Vec3d>(row, col)[dim];
					//vector.at<double>(dim + image_channels * (row + block_width * col)) = block.at<double>(row, col);

					cv::Mat values_of_layer1 = weight1 * vector;
					cv::Mat output_values = weight2 * values_of_layer1;

					cv::Mat recreated_block = cv::Mat(recreated_image,
						cv::Rect(j * block_width,
							i * block_height,
							block_width,
							block_height));

					for (int row = 0; row < block_height; row++)
						for (int col = 0; col < block_width; col++) {
							cv::Vec3d tmp_vector = cv::Vec3d();

							for (int dim = 0; dim < image_channels; dim++)
								tmp_vector[dim] = output_values.at<double>(dim + image_channels * (row + col));

							recreated_block.at<cv::Vec3d>(row, col) = tmp_vector;
						}
				}

			recreated_image = (recreated_image + 1) * 255.0 / 2;
			recreated_image.convertTo(recreated_image, CV_8UC3);

			std::filesystem::path output_directory{ resultPath + std::string("image") + std::to_string(k) + std::string(".png") };

			cv::imwrite(output_directory.u8string(), recreated_image);
		}
	}

	void forwardPropogation(const char* resultPath) { forwardPropogation(std::string(resultPath)); }

	void save() { ; }
};

int main()
{
	AutoencoderModel model = AutoencoderModel(256, 256, 3, 2, 2);
	model.sampleData("images");
	model.train(1);
	model.forwardPropogation("output\\");

	return 0;
}