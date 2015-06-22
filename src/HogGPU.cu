/*
 * HogGPU.cpp
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#include <include/HogGPU.h>
#include "HogGPU_impl.cuh"

#include <iostream>

#include <boost/thread.hpp>

#include <opencv2/gpu/gpu.hpp>

#include <include/Utils.h>

namespace ghog
{
namespace lib
{

HogGPU::HogGPU(std::string settings_file) :
	_settings(settings_file)
{
	_classifier = NULL;

	load_settings(settings_file);
}

HogGPU::~HogGPU()
{
// TODO Auto-generated destructor stub
}

void HogGPU::alloc_buffer(cv::Size buffer_size,
	int type,
	cv::Mat& buffer)
{
	cv::gpu::CudaMem cudamem(buffer_size.height, buffer_size.width, type,
		cv::gpu::CudaMem::ALLOC_ZEROCOPY);
	buffer = cudamem.createMatHeader();
	buffer.refcount = cudamem.refcount;
	buffer.addref();
	buffer.setTo(0);
}

GHOG_LIB_STATUS HogGPU::image_normalization(cv::Mat& image,
	ImageCallback* callback)
{
	boost::thread(&HogGPU::image_normalization_async, this, image, callback)
		.detach();
	return GHOG_LIB_STATUS_OK;
}

void HogGPU::image_normalization_async(cv::Mat& image,
	ImageCallback* callback)
{
	image_normalization_sync(image);
	callback->image_processed(image);
}

void HogGPU::image_normalization_sync(cv::Mat& image)
{
	//TODO
}

GHOG_LIB_STATUS HogGPU::calc_gradient(cv::Mat input_img,
	cv::Mat& magnitude,
	cv::Mat& phase,
	GradientCallback* callback)
{
	boost::thread(&HogGPU::calc_gradient_async, this, input_img, magnitude,
		phase, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void HogGPU::calc_gradient_async(cv::Mat input_img,
	cv::Mat& magnitude,
	cv::Mat& phase,
	GradientCallback* callback)
{
	calc_gradient_sync(input_img, magnitude, phase);
	callback->gradients_obtained(magnitude, phase);
}

void HogGPU::calc_gradient_sync(cv::Mat input_img,
	cv::Mat& magnitude,
	cv::Mat& phase)
{
	dim3 block_size(8, 8);
	dim3 grid_size;
	grid_size.x = input_img.cols / block_size.x;
	grid_size.y = input_img.rows / block_size.y;

	if(input_img.cols % block_size.x)
	{
		grid_size.x++;
	}
	if(input_img.rows % block_size.y)
	{
		grid_size.y++;
	}

	float* input_img_ptr = input_img.ptr< float >(0);
	float* magnitude_ptr = magnitude.ptr< float >(0);
	float* phase_ptr = phase.ptr< float >(0);

	float* device_input_img;
	float* device_magnitude;
	float* device_phase;

	cudaHostGetDevicePointer(&device_input_img, input_img_ptr, 0);
	cudaHostGetDevicePointer(&device_magnitude, magnitude_ptr, 0);
	cudaHostGetDevicePointer(&device_phase, phase_ptr, 0);

	gradient_kernel<<<grid_size, block_size>>>(device_input_img,
		device_magnitude, device_phase, input_img.rows, input_img.cols,
		input_img.step1(), magnitude.step1(), phase.step1());
	cudaDeviceSynchronize();
}

GHOG_LIB_STATUS HogGPU::create_descriptor(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat& descriptor,
	DescriptorCallback* callback)
{
	boost::thread(&HogGPU::create_descriptor_async, this, magnitude, phase,
		descriptor, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void HogGPU::create_descriptor_async(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat& descriptor,
	DescriptorCallback* callback)
{
	create_descriptor_sync(magnitude, phase, descriptor);
	callback->descriptor_obtained(descriptor);
}

void HogGPU::create_descriptor_sync(cv::Mat magnitude,
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

void HogGPU::calc_histogram(cv::Mat magnitude,
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

GHOG_LIB_STATUS HogGPU::classify(cv::Mat img,
	ClassifyCallback* callback)
{
	boost::thread(&HogGPU::classify_async, this, img, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void HogGPU::classify_async(cv::Mat img,
	ClassifyCallback* callback)
{
	callback->classification_result(classify_sync(img));
}

bool HogGPU::classify_sync(cv::Mat img)
{
	bool ret = false;
	cv::Mat resized;
	image_normalization_sync(img);
	cv::Mat grad_mag;
	cv::Mat grad_phase;
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

GHOG_LIB_STATUS HogGPU::locate(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride,
	LocateCallback* callback)
{
	boost::thread(&HogGPU::locate_async, this, img, roi, window_size,
		window_stride, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void HogGPU::locate_async(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride,
	LocateCallback* callback)
{
	callback->objects_located(
		locate_sync(img, roi, window_size, window_stride));
}

std::vector< cv::Rect > HogGPU::locate_sync(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride)
{
	std::vector< cv::Rect > ret;
	return ret;
}

void HogGPU::load_settings(std::string filename)
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

void HogGPU::set_classifier(IClassifier* classifier)
{
	_classifier = classifier;
}

GHOG_LIB_STATUS HogGPU::set_param(std::string param,
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

std::string HogGPU::get_param(std::string param)
{
	std::string module = get_module(param);
	if(module == "NULL")
	{
		return "Invalid parameter name.";
	} else
	{
		return _settings.load_str(module, param);
	}
}

std::string HogGPU::get_module(std::string param_name)
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

