/*
 * Hog.cpp
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#include <include/Hog.h>

#include <boost/thread.hpp>

namespace ghog
{
namespace lib
{

Hog::Hog(std::string settings_file) :
	_settings(settings_file)
{
	_classifier = NULL;

	_img_resize.width = _settings.load_int("Hog",
		"CLASSIFICATION_IMAGE_HEIGHT");
	_img_resize.width = _settings.load_int("Hog", "CLASSIFICATION_IMAGE_WIDTH");

	_num_bins = _settings.load_int(std::string("Descriptor"), "NUMBER_OF_BINS");
	_block_size.width = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_COLS");
	_block_size.height = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_ROWS");
}

Hog::~Hog()
{
// TODO Auto-generated destructor stub
}

GHOG_LIB_STATUS Hog::resize(cv::Mat image,
	cv::Size new_size,
	ImageCallback* callback)
{
	boost::thread(&Hog::resize_async, this, image, new_size, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS Hog::calc_gradient(cv::Mat input_img,
	ImageCallback* callback)
{
	boost::thread(&Hog::calc_gradient_async, this, input_img, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS Hog::create_descriptor(cv::Mat gradients,
	cv::Size block_size,
	int num_bins,
	DescriptorCallback* callback)
{
	boost::thread(&Hog::create_descriptor_async, this, gradients, block_size,
		num_bins, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS Hog::classify(cv::Mat img,
	ClassifyCallback* callback)
{
	boost::thread(&Hog::classify_async, this, img, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS Hog::locate(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride,
	LocateCallback* callback)
{
	boost::thread(&Hog::locate_async, this, img, roi, window_size,
		window_stride, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void Hog::set_classifier(IClassifier* classifier)
{
	_classifier = classifier;
}

void Hog::resize_async(cv::Mat image,
	cv::Size new_size,
	ImageCallback* callback)
{
	cv::Mat ret;
	resize_impl(image, new_size, ret);
	callback->image_processed(image, ret);
}

void Hog::calc_gradient_async(cv::Mat input_img,
	ImageCallback* callback)
{
	cv::Mat ret;
	calc_gradient_impl(input_img, ret);
	callback->image_processed(input_img, ret);
}

void Hog::create_descriptor_async(cv::Mat gradients,
	cv::Size block_size,
	int num_bins,
	DescriptorCallback* callback)
{
	cv::Mat ret;
	create_descriptor_impl(gradients, block_size, num_bins, ret);
	callback->descriptor_obtained(gradients, ret);
}

void Hog::classify_async(cv::Mat img,
	ClassifyCallback* callback)
{
	bool ret = false;
	cv::Mat gradients;
	resize_impl(img, _img_resize, gradients);
	calc_gradient_impl(gradients, gradients);
	cv::Mat descriptor;
	create_descriptor_impl(gradients, _block_size, _num_bins, descriptor);
	cv::Mat output = _classifier->classify(descriptor);
	if(output.at< float >(0) > 0)
	{
		ret = true;
	}
	callback->classification_result(img, ret);
}

void Hog::locate_async(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride,
	LocateCallback* callback)
{
	std::vector< cv::Rect > ret;
	callback->objects_located(img, ret);
}

void Hog::resize_impl(cv::Mat image,
	cv::Size new_size,
	cv::Mat& resized)
{
	cv::resize(image, resized, new_size, 0, 0, CV_INTER_AREA);
}

void Hog::calc_gradient_impl(cv::Mat input_img,
	cv::Mat& gradients)
{
	cv::Mat grad[2];
	cv::Sobel(input_img, grad[0], 1, 1, 0, 1);
	cv::Sobel(input_img, grad[1], 1, 0, 1, 1);
	cv::cartToPolar(grad[0], grad[1], grad[0], grad[1], true);
	cv::merge(grad, 2, gradients);
}

void Hog::create_descriptor_impl(cv::Mat gradients,
	cv::Size block_size,
	int num_bins,
	cv::Mat& descriptor)
{
	int total_cells = block_size.width * block_size.height;
	descriptor.create(total_cells, num_bins, CV_32FC1);
	int top_row = 0, bottom_row = 0, left_col = 0, right_col = 0;
	int row_step = gradients.rows / block_size.height;
	int extra_rows = gradients.rows % block_size.height;
	int col_step = gradients.cols / block_size.width;
	int extra_cols = gradients.cols % block_size.width;

	for(int i = 0; i < block_size.height; ++i)
	{
		bottom_row = top_row += row_step;
		if(extra_rows > 0)
		{
			extra_rows--;
			bottom_row++;
		}
		cv::Mat temp = gradients.rowRange(top_row, bottom_row);
		for(int j = 0; j < block_size.width; ++j)
		{
			right_col = left_col += col_step;
			if(extra_cols > 0)
			{
				extra_cols--;
				right_col++;
			}
			calc_histogram(temp.colRange(left_col, right_col), num_bins,
				descriptor.row(i * block_size.height + j));
			left_col = right_col;
		}
		extra_cols = gradients.cols % block_size.width;
		top_row = bottom_row;
	}

	descriptor.reshape(1, 1);
}

void Hog::calc_histogram(cv::Mat gradients,
	int num_bins,
	cv::Mat histogram)
{
	float bin_size = 360.0f / (float)num_bins;
	histogram = cv::Mat(1, num_bins, CV_32FC1, 0.0f);

	//TODO Split more efficiently (maybe use reshape to get only one channel)
	cv::Mat aux[2];
	cv::split(gradients, aux);
	cv::Mat mag = aux[0];
	cv::Mat phase = aux[1];

	int left_bin, right_bin;
	float delta;

	float mag_total = 0.0f;

	for(int i = 0; i < mag.rows; ++i)
	{
		for(int j = 0; j < mag.cols; ++j)
		{
			if(mag.at< float >(i, j) > 0)
			{
				left_bin = (int)floor(
					(phase.at< float >(i, j) - bin_size / 2.0f) / bin_size);
				if(left_bin < 0)
					left_bin += num_bins;
				right_bin = (left_bin + 1) % num_bins;

				delta = (phase.at< float >(i, j) / bin_size) - right_bin;
				if(delta > 1.0)
					delta -= num_bins;

				histogram.at< float >(left_bin) += (0.5 - delta)
					* mag.at< float >(i, j);
				histogram.at< float >(right_bin) += (0.5 + delta)
					* mag.at< float >(i, j);
				mag_total += mag.at< float >(i, j);
			}
		}
	}

	for(int i = 0; i < num_bins; ++i)
	{
		histogram.at< float >(i) /= mag_total;
	}
}

} /* namespace lib */
} /* namespace ghog */

