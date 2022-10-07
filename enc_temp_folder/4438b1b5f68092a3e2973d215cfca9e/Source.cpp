#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

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

int main()
{
	// Program settings
	bool is_trained = false;
	int epochs = 1;

	// Image parameters
	const int image_width = 256;
	const int image_height = 256;

	const int image_channels = 1;

	int image_type = CV_64FC1;
	int weight_type = CV_64FC1;

	// Learning parameters:
	const int block_width = 2;
	const int block_height = 2;

	const int block_area = block_width * block_height;
	const int block_volume = block_area * image_channels;

	// Initialization of model
	cv::Mat weight1 = cv::Mat(block_volume / 2, block_volume, weight_type);
	cv::randu(weight1, 0, 1);

	cv::Mat weight2 = cv::Mat(block_volume, block_volume / 2, weight_type);
	cv::randu(weight2, 0, 1);

	std::vector<cv::Mat> images;
	std::filesystem::path images_directory{ "images" };

	// training images loading
	for (auto& file : std::filesystem::directory_iterator{ images_directory })
	{
		std::string filename = file.path().u8string();

		cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

		image.convertTo(image, image_type);

		if (image.empty())
		{
			std::cerr << "error: can't open image.";
			return 1;
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

	// training
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		std::cout << "Epoch " << epoch + 1 << " =---------------------------------------=\n";
		for (int k = 0; k < images.size(); k++)
		{
			std::chrono::time_point start = std::chrono::system_clock::now();

			cv::Mat& image = images[k];

			// Dividing image by blocks

			std::vector<double> errors;

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
								//vector.at<double>(dim + image_channels * (row + block_width * col)) = block.at<cv::Vec3d>(row, col)[dim];
								vector.at<double>(dim + image_channels * (row + block_width * col)) = block.at<double>(row, col);

					cv::Mat values_of_layer1 = weight1 * vector;
					//std::cout << values_of_layer1;

					cv::Mat output_values = weight2 * values_of_layer1;

					cv::Mat delta = output_values - vector;

					cv::Mat tmp = delta.t() * delta;
					double error = tmp.at<double>(0, 0);

					errors.push_back(error);

					tmp = 1 / (values_of_layer1.t() * values_of_layer1);
					double learning_rate2 = tmp.at<double>(0, 0);

					tmp = 1 / (vector.t() * vector);
					double learning_rate1 = tmp.at<double>(0, 0);

					//learning_rate1 = 0.005;
					//learning_rate2 = 0.005;

					weight2 = weight2 - learning_rate2 * delta * values_of_layer1.t();
					weight1 = weight1 - learning_rate1 * weight2.t() * delta * vector.t();

					//weight2 /= weight2.cols;
					//weight1 /= weight1.cols;

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

			std::cout << "Processing image(" << k + 1 << "/" << images.size() << "): " << std::to_string(sum_error)
				<< " time left: " << minutes << "m " << seconds << "s" << "\n";
		}
	}

	if (is_trained)
	{
		weight1 = cv::imread("..\\weight1.png", weight_type);
		weight2 = cv::imread("..\\weight2.png", weight_type);
	}

	else
	{
		cv::imwrite("weight1.png", weight1);
		cv::imwrite("weight2.png", weight2);
	}

	// Forward propagation of trained model
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
							//vector.at<double>(dim + image_channels * (row + block_width * col)) = block.at<cv::Vec3d>(row, col)[dim];
							vector.at<double>(dim + image_channels * (row + block_width * col)) = block.at<double>(row, col);

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
							tmp_vector[dim] = output_values.at<double>(dim + image_channels * (row + block_width * col));
								
						recreated_block.at<cv::Vec3d>(row, col) = tmp_vector;
					}
			}

		recreated_image = (recreated_image + 1) * 255.0 / 2;

		recreated_image.convertTo(recreated_image, CV_8UC3);

		std::filesystem::path output_directory{ std::string("./output/image") + std::to_string(k) + std::string(".png")};

		cv::imwrite(output_directory.u8string(), recreated_image);
	}

	return 0;
}