#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>
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
	// Initialization of model
	std::vector<cv::Mat> images;
	std::filesystem::path images_directory{ "images" };	

	// Learning parameters: Image size is 256x256
	const int image_width = 256;
	const int image_height = 256;

	const int image_channels = 3;

	const int block_width = 64;
	const int block_height = 64;

	const int block_area = block_width * block_height;

	for (auto& file : std::filesystem::directory_iterator{ images_directory })
	{
		std::string filename = file.path().u8string();

		images.push_back(cv::imread(filename, cv::IMREAD_COLOR));

		images.back().convertTo(images.back(), CV_32FC3);

		if (images.back().empty())
		{
			std::cerr << "error: can't open image.";
			return 1;
		}

		images.back() = images.back() * 2 / 255.0 - 1; // Regularize image

		std::vector<cv::Mat> vectors;
		// Dividing image by blocks
		for (int i = 0; i < image_height / block_height; i ++)
			for (int j = 0; j < image_width / block_width; j++)
			{
				cv::Mat block = cv::Mat(images.back(),
					cv::Rect(j * block_width,
						i * block_height,
						(j + 1) * block_width - j * block_width,
						(i + 1) * block_height - i * block_height));

				//std::cout << j * block_width << " " << i * block_height << " " << (j + 1) * block_width - 1 << " " << (i + 1) * block_height - 1 << "\n";
				
				/*cv::Mat block = cv::Mat(images.back(),
					cv::Rect(64,
						0,
						4,
						4));

				std::cout << images.back().at<cv::Vec3f>(127, 63)[0] << "\n";

				std::cout << block << "\n";

				std::cout << block.at<cv::Vec3f>(3, 3)[0] << "\n";*/

				// Working with distinct blocks
			/*	std::cout << block.size().height << "\n";
				std::cout << block.size().width << "\n";*/

				//std::cout << block.at<cv::Vec3f>(63, 63)[0] << "\n";

				for (int row = 0; row < block_height; row ++)
					for (int col = 0; col < block_width; col++)
					{
						for (int dim = 0; dim < image_channels; dim++)
						{

							cv::Mat vector = cv::Mat(block_area * image_channels, 1, CV_32F);

							vector.at<float>(dim * block_area + row * block_height + col) = block.at<cv::Vec3f>(row, col)[dim];
						}
					}
			}
	}


	return 0;
}