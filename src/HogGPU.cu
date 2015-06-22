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
	HogDescriptor(settings_file)
{

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

void HogDescriptor::normalize_blocks(cv::Mat& descriptor)
{
	int cells_per_block = _block_size.height * _block_size.width;
	int elements_per_block = cells_per_block * _num_bins;

	for(int i = 0; i < descriptor.cols; i += elements_per_block)
	{
		float L1_norm = 0.0f;
		for(int j = 0; j < elements_per_block; ++j)
		{
			L1_norm += descriptor.at< float >(i + j);
		}
		for(int j = 0; j < elements_per_block; ++j)
		{
			descriptor.at< float >(i + j) = sqrt(
				descriptor.at< float >(i + j) / L1_norm);
		}
	}
}

//GHOG_LIB_STATUS HogGPU::classify(cv::Mat img,
//	ClassifyCallback* callback)
//{
//	boost::thread(&HogGPU::classify_async, this, img, callback).detach();
//	return GHOG_LIB_STATUS_OK;
//}
//
//bool HogGPU::classify_sync(cv::Mat img)
//{
//	bool ret = false;
//	cv::Mat resized;
//	image_normalization_sync(img);
//	cv::Mat grad_mag;
//	cv::Mat grad_phase;
//	calc_gradient_sync(img, grad_mag, grad_phase);
//	cv::Mat descriptor;
//	create_descriptor_sync(grad_mag, grad_phase, descriptor);
//	cv::Mat output = _classifier->classify_sync(descriptor);
//	if(output.at< float >(0) > 0)
//	{
//		ret = true;
//	}
//	return ret;
//}
//
//GHOG_LIB_STATUS HogGPU::locate(cv::Mat img,
//	cv::Rect roi,
//	cv::Size window_size,
//	cv::Size window_stride,
//	LocateCallback* callback)
//{
//	boost::thread(&HogGPU::locate_async, this, img, roi, window_size,
//		window_stride, callback).detach();
//	return GHOG_LIB_STATUS_OK;
//}
//
//std::vector< cv::Rect > HogGPU::locate_sync(cv::Mat img,
//	cv::Rect roi,
//	cv::Size window_size,
//	cv::Size window_stride)
//{
//	std::vector< cv::Rect > ret;
//	return ret;
//}

} /* namespace lib */
} /* namespace ghog */

