/*
 * HogCPU.cpp
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#include <include/HogCPU.h>

#include <iostream>

#include <boost/thread.hpp>

#include <opencv2/imgproc/imgproc.hpp>

namespace ghog
{
namespace lib
{

HogCPU::HogCPU(std::string settings_file) :
	_settings(settings_file)
{
	_classifier = NULL;

	load_settings(settings_file);
}

HogCPU::~HogCPU()
{
// TODO Auto-generated destructor stub
}

cv::Mat HogCPU::alloc_buffer(cv::Size buffer_size,
	int type,
	int border_size)
{
	//Allocate extra space for the borders. Force output to zero.
	cv::Mat buf(buffer_size.height + 2 * border_size,
		buffer_size.width + 2 * border_size, type, 0);
	//Return the matrix without the borders
	//The methods rowRange and colRange are 0-indexed, inclusive on the first
	//parameter and exclusive on the second.
	return buf.rowRange(border_size, buf.rows - border_size).colRange(
		border_size, buf.cols - border_size);
}

GHOG_LIB_STATUS HogCPU::resize(cv::Mat image,
	cv::Size new_size,
	cv::Mat& resized_image,
	ImageCallback* callback)
{
	boost::thread(&HogCPU::resize_async, this, image, new_size, resized_image,
		callback).detach();
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS HogCPU::calc_gradient(cv::Mat input_img,
	cv::Mat& magnitude,
	cv::Mat& phase,
	GradientCallback* callback)
{
	boost::thread(&HogCPU::calc_gradient_async, this, input_img, magnitude,
		phase, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS HogCPU::create_descriptor(cv::Mat gradients,
	cv::Size block_size,
	int num_bins,
	DescriptorCallback* callback)
{
	boost::thread(&HogCPU::create_descriptor_async, this, gradients, block_size,
		num_bins, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS HogCPU::classify(cv::Mat img,
	ClassifyCallback* callback)
{
	boost::thread(&HogCPU::classify_async, this, img, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS HogCPU::locate(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride,
	LocateCallback* callback)
{
	boost::thread(&HogCPU::locate_async, this, img, roi, window_size,
		window_stride, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void HogCPU::load_settings(std::string filename)
{
	_img_resize.width = _settings.load_int("Hog",
		"CLASSIFICATION_IMAGE_HEIGHT");
	_img_resize.width = _settings.load_int("Hog", "CLASSIFICATION_IMAGE_WIDTH");

	_num_bins = _settings.load_int(std::string("Descriptor"), "NUMBER_OF_BINS");
	_block_size.width = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_COLS");
	_block_size.height = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_ROWS");
}

void HogCPU::set_classifier(IClassifier* classifier)
{
	_classifier = classifier;
}

GHOG_LIB_STATUS HogCPU::set_img_resize(cv::Size img_resize)
{
	_img_resize = img_resize;
	return GHOG_LIB_STATUS_OK;
}

cv::Size HogCPU::get_img_resize()
{
	return _img_resize;
}

GHOG_LIB_STATUS HogCPU::set_num_bins(int num_bins)
{
	_num_bins = num_bins;
	return GHOG_LIB_STATUS_OK;
}

int HogCPU::get_num_bins()
{
	return _num_bins;
}

GHOG_LIB_STATUS HogCPU::set_block_size(cv::Size block_size)
{
	_block_size = block_size;
	return GHOG_LIB_STATUS_OK;
}

cv::Size HogCPU::get_block_size()
{
	return _block_size;
}

void HogCPU::resize_async(cv::Mat image,
	cv::Size new_size,
	cv::Mat& resized_image,
	ImageCallback* callback)
{
	resize_impl(image, new_size, resized_image);
	callback->image_processed(image, resized_image);
}

void HogCPU::calc_gradient_async(cv::Mat input_img,
	cv::Mat& magnitude,
	cv::Mat& phase,
	GradientCallback* callback)
{
	calc_gradient_impl(input_img, magnitude, phase);
	callback->gradients_obtained(input_img, magnitude, phase);
}

void HogCPU::create_descriptor_async(cv::Mat gradients,
	cv::Size block_size,
	int num_bins,
	DescriptorCallback* callback)
{
	cv::Mat ret;
	create_descriptor_impl(gradients, block_size, num_bins, ret);
	callback->descriptor_obtained(gradients, ret);
}

void HogCPU::classify_async(cv::Mat img,
	ClassifyCallback* callback)
{
	bool ret = false;
	cv::Mat resized;
	resize_impl(img, _img_resize, resized);
	cv::Mat grad_mag;
	cv::Mat grad_phase;
	calc_gradient_impl(img, grad_mag, grad_phase);
	cv::Mat descriptor;
	create_descriptor_impl(resized, _block_size, _num_bins, descriptor);
	cv::Mat output = _classifier->classify_sync(descriptor);
	if(output.at< float >(0) > 0)
	{
		ret = true;
	}
	callback->classification_result(img, ret);
}

void HogCPU::locate_async(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride,
	LocateCallback* callback)
{
	std::vector< cv::Rect > ret;
	callback->objects_located(img, ret);
}

void HogCPU::resize_impl(cv::Mat image,
	cv::Size new_size,
	cv::Mat& resized_image)
{
	cv::resize(image, resized_image, new_size, 0, 0, CV_INTER_LINEAR);
}

void HogCPU::calc_gradient_impl(cv::Mat input_img,
	cv::Mat& magnitude,
	cv::Mat& phase)
{
	//Store dx temporarily on magnitude matrix
	cv::Sobel(input_img, magnitude, -1, 1, 0, 1);
	//Store dy temporarily on phase matrix
	cv::Sobel(input_img, phase, -1, 0, 1, 1);
	cv::cartToPolar(magnitude, phase, magnitude, phase, true);
}

void HogCPU::create_descriptor_impl(cv::Mat gradients,
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

void HogCPU::calc_histogram(cv::Mat gradients,
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
				if(right_bin == 0)
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

