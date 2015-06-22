/*
 * HogCPU.cpp
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#include <include/HogCPU.h>

#include <boost/thread.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#include <include/Utils.h>

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

void HogCPU::alloc_buffer(cv::Size buffer_size,
	int type,
	cv::Mat& buffer)
{
	buffer.create(buffer_size.height, buffer_size.width, type);
	buffer.setTo(0);
}

GHOG_LIB_STATUS HogCPU::image_normalization(cv::Mat& image,
	ImageCallback* callback)
{
	boost::thread(&HogCPU::image_normalization_async, this, image, callback)
		.detach();
	return GHOG_LIB_STATUS_OK;
}

void HogCPU::image_normalization_async(cv::Mat& image,
	ImageCallback* callback)
{
	image_normalization_sync(image);
	callback->image_processed(image);
}

void HogCPU::image_normalization_sync(cv::Mat& image)
{
	//TODO
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

void HogCPU::calc_gradient_async(cv::Mat input_img,
	cv::Mat& magnitude,
	cv::Mat& phase,
	GradientCallback* callback)
{
	calc_gradient_sync(input_img, magnitude, phase);
	callback->gradients_obtained(magnitude, phase);
}

void HogCPU::calc_gradient_sync(cv::Mat input_img,
	cv::Mat& magnitude,
	cv::Mat& phase)
{
	//Store dx temporarily on magnitude matrix
	cv::Sobel(input_img, magnitude, -1, 1, 0, 1);
	//Store dy temporarily on phase matrix
	cv::Sobel(input_img, phase, -1, 0, 1, 1);
	cv::cartToPolar(magnitude, phase, magnitude, phase, true);
}

GHOG_LIB_STATUS HogCPU::create_descriptor(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat& descriptor,
	DescriptorCallback* callback)
{
	boost::thread(&HogCPU::create_descriptor_async, this, magnitude, phase,
		descriptor, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void HogCPU::create_descriptor_async(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat& descriptor,
	DescriptorCallback* callback)
{
	create_descriptor_sync(magnitude, phase, descriptor);
	callback->descriptor_obtained(descriptor);
}

void HogCPU::create_descriptor_sync(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat& descriptor)
{
	//TODO: verify that magnitude and phase have same size and type.
	//TODO: get preallocated descriptor and verify it, instead of creating.

	cv::Size cell_grid = Utils::partition(magnitude.size(), _cell_size);
	int total_cells = cell_grid.width * cell_grid.height;
	int blocks_per_cell = _block_size.width * _block_size.height;
	int total_outputs = total_cells * blocks_per_cell * _num_bins;
	descriptor.create(1, total_outputs, CV_32FC1);
	cv::Mat histograms(total_cells, _num_bins, CV_32FC1);
	int top_row = 0, bottom_row = 0, left_col = 0, right_col = 0;
	int extra_rows = magnitude.rows % cell_grid.height;
	int extra_cols = magnitude.cols % cell_grid.width;
	int output_col_left = 0;
	int output_col_right = _num_bins;
	int histogram_row = 0;

	for(int i = 0; i < cell_grid.height; ++i)
	{
		bottom_row = top_row += _cell_size.height;
		if(extra_rows > 0)
		{
			extra_rows--;
			bottom_row++;
		}
		cv::Mat mag_aux = magnitude.rowRange(top_row, bottom_row);
		cv::Mat phase_aux = phase.rowRange(top_row, bottom_row);
		for(int j = 0; j < cell_grid.width; ++j)
		{
			right_col = left_col += _cell_size.width;
			if(extra_cols > 0)
			{
				extra_cols--;
				right_col++;
			}
			calc_histogram(mag_aux.colRange(left_col, right_col),
				phase_aux.colRange(left_col, right_col),
				histograms.row(histogram_row));
			histogram_row++;
			left_col = right_col;
		}
		extra_cols = magnitude.cols % cell_grid.width;
		top_row = bottom_row;
	}

	//TODO: normalize and put on the descriptor
}

void HogCPU::calc_histogram(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat histogram)
{
	float bin_size = 360.0f / (float)_num_bins;

	int left_bin, right_bin;
	float delta;

	float mag_total = 0.0f;

	for(int i = 0; i < magnitude.rows; ++i)
	{
		for(int j = 0; j < magnitude.cols; ++j)
		{
			if(magnitude.at< float >(i, j) > 0)
			{
				left_bin = (int)floor(
					(phase.at< float >(i, j) - bin_size / 2.0f) / bin_size);
				if(left_bin < 0)
					left_bin += _num_bins;
				right_bin = (left_bin + 1) % _num_bins;

				delta = (phase.at< float >(i, j) / bin_size) - right_bin;
				if(right_bin == 0)
					delta -= _num_bins;

				histogram.at< float >(left_bin) += (0.5 - delta)
					* magnitude.at< float >(i, j);
				histogram.at< float >(right_bin) += (0.5 + delta)
					* magnitude.at< float >(i, j);
				mag_total += magnitude.at< float >(i, j);
			}
		}
	}

	for(int i = 0; i < _num_bins; ++i)
	{
		histogram.at< float >(i) /= mag_total;
	}
}

GHOG_LIB_STATUS HogCPU::classify(cv::Mat img,
	ClassifyCallback* callback)
{
	boost::thread(&HogCPU::classify_async, this, img, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void HogCPU::classify_async(cv::Mat img,
	ClassifyCallback* callback)
{
	bool ret = classify_sync(img);
	callback->classification_result(ret);
}

bool HogCPU::classify_sync(cv::Mat img)
{
	bool ret = false;
	image_normalization_sync(img);
	cv::Mat grad_mag(img.size(), img.type());
	cv::Mat grad_phase(img.size(), img.type());
	calc_gradient_sync(img, grad_mag, grad_phase);
	cv::Mat descriptor;
	create_descriptor_sync(grad_mag, grad_phase, descriptor);
	cv::Mat output = _classifier->classify_sync(descriptor);
	if(output.at< float >(0) > 0)
	{
		ret = true;
	}
	return ret;
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

void HogCPU::locate_async(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride,
	LocateCallback* callback)
{
	std::vector< cv::Rect > ret = locate_sync(img, roi, window_size,
		window_stride);
	callback->objects_located(ret);
}

std::vector< cv::Rect > HogCPU::locate_sync(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride)
{
	std::vector< cv::Rect > ret;
	return ret;
}

void HogCPU::load_settings(std::string filename)
{
	_num_bins = _settings.load_int(std::string("Descriptor"), "NUMBER_OF_BINS");
	_block_size.width = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_COLS");
	_block_size.height = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_ROWS");
	_cell_size.width = _settings.load_int(std::string("Descriptor"),
		"CELL_SIZE_COLS");
	_cell_size.height = _settings.load_int(std::string("Descriptor"),
		"CELL_SIZE_ROWS");
}

void HogCPU::set_classifier(IClassifier* classifier)
{
	_classifier = classifier;
}

GHOG_LIB_STATUS HogCPU::set_param(std::string param,
	std::string value)
{
	std::string module = get_module(param);
	if(module == "NULL")
	{
		return GHOG_LIB_STATUS_INVALID_PARAMETER_NAME;
	}
	_settings.save(module, param, value.c_str());
	return GHOG_LIB_STATUS_OK;
}

std::string HogCPU::get_param(std::string param)
{
	std::string module = get_module(param);
	if (module == "NULL")
	{
		return "Invalid parameter name.";
	} else
	{
		return _settings.load_str(module, param);
	}
}

std::string HogCPU::get_module(std::string param_name)
{
	if((param_name == "CELL_SIZE_COLS") || (param_name == "CELL_SIZE_ROWS")
		|| (param_name == "BLOCK_SIZE_COLS")
		|| (param_name == "BLOCK_SIZE_ROWS")
		|| (param_name == "NUMBER_OF_BINS"))
	{
		return "Descriptor";
	} else if((param_name == "TYPE") || (param_name == "FILENAME"))
	{
		return "Classifier";
	} else
	{
		return "NULL";
	}
}

} /* namespace lib */
} /* namespace ghog */

