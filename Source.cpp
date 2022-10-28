/*
* Лабораторная работа №1 по предмету МРЗИС на тему "Сжатие графической информации линейной рециркуляционной сетью."
* Выполнил: Артеменко Дмитрий, студент БГУИР группа 021731
* 
* Сторонние библиотеки:
* - OpenCV2: https://opencv.org/  - - - - - - - - Официальный сайт библиотеки
*			 https://github.com/opencv/opencv - - Репозиторий библиотеки
*
*/

#include <iostream>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <iomanip>
#include <vector>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>

class AutoencoderModel
{
private:
	const int image_type = CV_64FC3;
	const int weight_type = CV_64FC1;

	// Веса
	cv::Mat weight1;
	cv::Mat weight2;

	int code_size; // Число нейронов кодового слоя

	// Параметры блока
	int block_width;
	int block_height;
	int block_area;
	int block_volume;

	int block_count;

	int row_blocks;
	int col_blocks;
	int overflow; // Индикатор выхода блока за пределы изображения

	// Параметры изображения
	int image_width;
	int image_height;
	int image_channels;

	double acceptableError;

	// Тренировочные данные
	std::vector<cv::Mat> images;

public:
	AutoencoderModel(int image_width, int image_height, int image_channels, int block_width,
		int block_height, int code_size, double error)
		: image_width(image_width), image_height(image_height), image_channels(image_channels),
		block_width(block_width), block_height(block_height), block_area(block_width* block_height),
		block_volume(block_area* image_channels), block_count(0), code_size(code_size), acceptableError(error)
	{
		weight1 = cv::Mat(code_size, block_volume, weight_type); // Начальная инициализация весов
		cv::randu(weight1, -1, 1);

		weight2 = cv::Mat(block_volume, code_size, weight_type);
		cv::randu(weight2, -1, 1);

		overflow = image_height % block_height;

		row_blocks = image_height / block_height;
		if (overflow > 0)
			row_blocks += 1;

		col_blocks = image_width / block_width;
		if (overflow > 0)
			col_blocks += 1;

		block_count = row_blocks * col_blocks;
	}

	void saveImage(cv::Mat image, std::string name)
	{
		cv::Mat tmp = image.clone();
		tmp = (tmp + 1) / 2.0 * 255;
		tmp.convertTo(tmp, CV_8UC1);

		cv::imwrite(name, tmp);
	}

	void saveImage(cv::Mat image, char* name) { saveImage(image, std::string(name)); }

	cv::Mat loadImage(std::string name)
	{
		weight1 = cv::Mat(cv::imread(name, cv::IMREAD_UNCHANGED));
		weight1.convertTo(weight1, weight_type);

		weight1 = weight1 / 255.0 * 2 - 1;

		return weight1;
	}

	cv::Mat loadImage(char* name) { return loadImage(std::string(name)); }
	
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

	std::vector<cv::Mat> divideImageByBlocks(cv::Mat image)
	{
		std::vector<cv::Mat> vectors;

		for (int i = 0; i < row_blocks; i++)
		{
			int width_offset = 0;
			int height_offset = 0;

			for (int j = 0; j < col_blocks; j++)
			{
				if (i == row_blocks - 1)
					height_offset = block_height - overflow;

				if (j == col_blocks - 1)
					width_offset = block_width - overflow;

				cv::Mat block = cv::Mat(image,
					cv::Rect(j * block_width - width_offset,
						i * block_height - height_offset,
						block_width,
						block_height));

				cv::Mat vector = cv::Mat(block_area * image_channels, 1, weight_type);

				for (int row = 0; row < block_height; row++)
					for (int col = 0; col < block_width; col++)
						for (int dim = 0; dim < image_channels; dim++)
							vector.at<double>(dim + image_channels * col + image_channels * block_width * row) = block.at<cv::Vec3d>(row, col)[dim];

				vectors.push_back(vector);
			}
		}

		return vectors;
	}

	cv::Mat restoreFromVectors(std::vector<cv::Mat> vectors)
	{
		cv::Mat recreated_image = cv::Mat(image_width, image_height, image_type);

		int count = 0;

		for (cv::Mat vector : vectors)
		{
			int row_block = count / block_height;
			int col_block = count % block_height;

			int height_offset = 0;
			int width_offset = 0;

			if (row_block == row_blocks - 1)
				height_offset = block_height - overflow;

			if (col_block == col_blocks - 1)
				width_offset = block_width - overflow;

			cv::Mat recreated_block = cv::Mat(recreated_image,
				cv::Rect(col_block * block_width - width_offset,
					row_block * block_height - height_offset,
					block_width,
					block_height));

			for (int row = 0; row < row_block; row++)
				for (int col = 0; col < col_block; col++) {
					cv::Vec3d tmp_vector = cv::Vec3d();

					for (int dim = 0; dim < image_channels; dim++)
						tmp_vector[dim] = vector.at<double>(dim + image_channels * col + image_channels * block_width * row);

					recreated_block.at<cv::Vec3d>(row, col) = tmp_vector;
				}
		}

		return recreated_image;
	}

	void train()
	{
		int iterationCount = 0;
		double trainSetError = 0;

		do
		{
			std::vector<cv::Mat> vectors = divideImageByBlocks(images[0]);

			std::vector<double> perVectorError;

			for (cv::Mat vector: vectors)
			{
				cv::Mat values_of_layer1 = weight1 * vector;
				cv::Mat output_values = weight2 * values_of_layer1;

				cv::Mat delta = output_values - vector;

				cv::Mat tmp = delta.t() * delta;
				double error = tmp.at<double>(0, 0);
				perVectorError.push_back(error);

				// ++++ Адаптивный параметр обучения +++++++++++++++++++
				tmp = 1 / (values_of_layer1.t() * values_of_layer1);
				double learning_rate2 = tmp.at<double>(0, 0);

				tmp = 1 / (vector.t() * vector);
				double learning_rate1 = tmp.at<double>(0, 0);
				// +++++++++++++++++++++++++++++++++++++++++++++++++++++

				cv::Mat previousIterationWeight2 = weight2.clone();

				weight2 = weight2 - learning_rate2 * delta * values_of_layer1.t();
				weight1 = weight1 - learning_rate1 * previousIterationWeight2.t() * delta * vector.t();
			}

			trainSetError = 0;
			for (int i = 0; i < perVectorError.size(); i++)
				trainSetError += perVectorError[i];

			std::cout << "Iteration #" << ++iterationCount << " | error: " << trainSetError << "\n";
		} 
		while (trainSetError > acceptableError);
	}

	cv::Mat codeImage()
	{
		std::vector<cv::Mat> vectors = divideImageByBlocks(images[0]);
		
		cv::Mat codedImage = cv::Mat(vectors[0].rows, vectors.size(), weight_type);

		int col = 0;

		for (cv::Mat vector: vectors)
		{
			cv::Mat codedVector = weight1 * vector;

			for (int row = 0; row < codedVector.size[0]; row++)
			{
				codedImage.at<cv::Scalar>(row, col) = codedVector.at<cv::Scalar>(row, col);
			}

			col++;
		}

		return codedImage;
	}

	cv::Mat decodeImage(cv::Mat image)
	{
		std::vector<cv::Mat> vectors;
		for (int i = 0; i < image.cols; i++)
		{
			cv::Mat vector = cv::Mat(image.rows, 1, weight_type);

			for (int j = 0; j < image.rows; j++)
				vector.at<cv::Scalar>(j, 0) = image.at<cv::Scalar>(j, i);

			vectors.push_back(vector);
		}

		std::vector<cv::Mat> outputVectors;
		for (cv::Mat vector : vectors)
		{
			cv::Mat output_vector = weight2 * vector;
			outputVectors.push_back(output_vector);
		}

		cv::Mat decodedImage = restoreFromVectors(outputVectors);

		return decodedImage;
	}

	/*void forwardPropogation(std::string resultPath = "output")
	{
		for (int k = 0; k < images.size(); k++)
		{
			cv::Mat image = images[k];

			cv::Mat recreated_image = cv::Mat(image_width, image_height, image_type); // Output result of neural network

			for (int i = 0; i < row_blocks; i++)
			{
				int width_offset = 0;
				int height_offset = 0;

				for (int j = 0; j < col_blocks; j++)
				{
					if (i == row_blocks - 1)
						height_offset = block_height - overflow;

						if (j == col_blocks - 1)
							width_offset = block_width - overflow;

						cv::Mat block = cv::Mat(image,
							cv::Rect(j * block_width - width_offset,
								i * block_height - height_offset,
								block_width,
								block_height));


					//cv::Mat block = cv::Mat(image,
					//	cv::Rect(j * block_width,
					//		i * block_height,
					//		block_width,
					//		block_height));

					cv::Mat vector = cv::Mat(block_volume, 1, weight_type);

					for (int row = 0; row < block_height; row++)
						for (int col = 0; col < block_width; col++)
							for (int dim = 0; dim < image_channels; dim++)
								vector.at<double>(dim + image_channels * col + image_channels * block_width * row) = block.at<cv::Vec3d>(row, col)[dim];
					//vector.at<double>(dim + image_channels * (row + block_width * col)) = block.at<double>(row, col);

					cv::Mat values_of_layer1 = weight1 * vector;
					cv::Mat output_values = weight2 * values_of_layer1;

					cv::Mat recreated_block = cv::Mat(recreated_image,
						cv::Rect(j * block_width - width_offset,
							i * block_height - height_offset,
							block_width,
							block_height));

					for (int row = 0; row < block_height; row++)
						for (int col = 0; col < block_width; col++) {
							cv::Vec3d tmp_vector = cv::Vec3d();

							for (int dim = 0; dim < image_channels; dim++)
								tmp_vector[dim] = output_values.at<double>(dim + image_channels * col + image_channels * block_width * row);

							recreated_block.at<cv::Vec3d>(row, col) = tmp_vector;
						}
				}
			}

			recreated_image = (recreated_image + 1) * 255.0 / 2;
			recreated_image.convertTo(recreated_image, CV_8UC3);

			std::filesystem::path output_directory{ resultPath + std::string("image") + std::to_string(k) + std::string(".jpg") };

			cv::imwrite(output_directory.u8string(), recreated_image);
		}
	}*/

	/*void forwardPropogation(const char* resultPath) { forwardPropogation(std::string(resultPath)); }
	*/
	double compressionCoefficient() { return double(block_volume) * block_count / ((block_volume + block_count) * code_size + 2); }

	void saveModel()
	{
		saveImage(weight1, "weight1.jpg");
		saveImage(weight2, "weight2.jpg");
	}

	void loadModel()
	{
		weight1 = loadImage("weight1.jpg");
		weight2 = loadImage("weight2.jpg");
	}

	/*void save()
	{ 
		cv::Mat tmp = weight1.clone();
		tmp = (tmp + 1) / 2.0 * 255;
		tmp.convertTo(tmp, CV_8UC1);

		cv::imwrite("weight1.jpg", tmp);

		tmp = weight2.clone();
		tmp = (tmp + 1) / 2.0 * 255;
		tmp.convertTo(tmp, CV_8UC1);

		cv::imwrite("weight2.jpg", tmp);
	}

	void load()
	{
		weight1 = cv::Mat(cv::imread("weight1.jpg", cv::IMREAD_UNCHANGED));
		weight1.convertTo(weight1, weight_type);

		weight1 = weight1 / 255.0 * 2 - 1;

		weight2 = cv::Mat(cv::imread("weight2.jpg", cv::IMREAD_UNCHANGED));
		weight2.convertTo(weight2, weight_type);

		weight2 = weight2 / 255.0 * 2 - 1;
	}*/
};

int main()
{
	// --------------------------------------| W |  H|CHAN|BW| BH |CSZ|
	AutoencoderModel model = AutoencoderModel(256, 256, 3, 4, 4, 10, 1);

	model.sampleData("images");
	model.train();
	model.saveModel();

	cv::Mat coded = model.codeImage();
	model.saveImage(coded, "output\\coded.jpg");

	cv::Mat decoded = model.decodeImage(coded);
	model.saveImage(decoded, "output\\decoded.jpg");

	return 0;
}