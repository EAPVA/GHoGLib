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
	cv::Mat& buffer,
	int padding_size)
{
	cv::gpu::CudaMem cudamem(buffer_size.height + 2 * padding_size,
		buffer_size.width + 2 * padding_size, type,
		cv::gpu::CudaMem::ALLOC_ZEROCOPY);
	buffer = cudamem.createMatHeader().rowRange(padding_size,
		cudamem.rows - padding_size).colRange(padding_size,
		cudamem.cols - padding_size);
	buffer.refcount = cudamem.refcount;
	buffer.addref();
	cv::Mat header_temp = cudamem.createMatHeader();
	header_temp.setTo(0);
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
	dim3 block_size(3, 64, 1);
	dim3 grid_size;
	grid_size.x = image.cols / block_size.y;
	grid_size.y = image.rows / block_size.z;

	if(image.cols % block_size.y)
	{
		grid_size.x++;
	}
	if(image.rows % block_size.z)
	{
		grid_size.y++;
	}

	float* input_img_ptr = image.ptr< float >(0);
	float* device_input_img;
	cudaHostGetDevicePointer(&device_input_img, input_img_ptr, 0);

	gamma_norm_kernel<<<grid_size, block_size>>>(device_input_img, image.rows,
		image.cols, image.step1());
	cudaDeviceSynchronize();
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
	dim3 block_size(3, 64, 1);
	dim3 grid_size;
	grid_size.x = input_img.cols / block_size.y;
	grid_size.y = input_img.rows / block_size.z;

	if(input_img.cols % block_size.y)
	{
		grid_size.x++;
	}
	if(input_img.rows % block_size.z)
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

void HogGPU::create_descriptor_sync(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat& descriptor,
	cv::Mat& histograms)
{
	//TODO: verify that magnitude and phase have correct size and type.
	//TODO: verify that the descriptor has correct size and type
	//TODO: possibly preallocate histograms auxiliary matrix

	cv::Size hog_block_grid(
		((_cell_grid.width - _block_size.width) / _block_stride.width) + 1,
		((_cell_grid.height - _block_size.height) / _block_stride.height) + 1);

	dim3 block_size_hist(64, 1);
	dim3 grid_size;
	grid_size.x = magnitude.cols / block_size_hist.x;
	grid_size.y = _cell_grid.height / block_size_hist.y;

	if(magnitude.cols % block_size_hist.x)
	{
		grid_size.x++;
	}
	if(_cell_grid.height % block_size_hist.y)
	{
		grid_size.y++;
	}

	float* magnitude_ptr = magnitude.ptr< float >(0);
	float* phase_ptr = phase.ptr< float >(0);
	float* descriptor_ptr = descriptor.ptr< float >(0);
	float* histograms_ptr = histograms.ptr< float >(0);

	float* device_magnitude;
	float* device_phase;
	float* device_descriptor;
	float* device_histograms;

	cudaHostGetDevicePointer(&device_magnitude, magnitude_ptr, 0);
	cudaHostGetDevicePointer(&device_phase, phase_ptr, 0);
	cudaHostGetDevicePointer(&device_descriptor, descriptor_ptr, 0);
	cudaHostGetDevicePointer(&device_histograms, histograms_ptr, 0);

//	int cell_row_step = _cell_grid.width * _num_bins;

	histogram_kernel<<<grid_size, block_size_hist>>>(device_magnitude,
		device_phase, device_histograms, magnitude.cols, magnitude.rows,
		_cell_grid.width, _cell_grid.height, magnitude.step1(), phase.step1(),
		histograms.step1(), _cell_size.width, _cell_size.height, _num_bins);

	dim3 block_size_norm(9, 4, 8);

	grid_size.x = hog_block_grid.width / 8; // 8 hog blocks per thread block
	grid_size.y = hog_block_grid.height;

	if(hog_block_grid.width % 8)
	{
		grid_size.x++;
	}

	cudaDeviceSynchronize();

	block_normalization_kernel<<<grid_size, block_size_norm>>>(
		device_histograms, device_descriptor, histograms.step1(),
		hog_block_grid.width, hog_block_grid.height, _block_size.width,
		_block_size.height, _num_bins, _cell_grid.width, _block_stride.width,
		_block_stride.height);
	cudaDeviceSynchronize();
}

} /* namespace lib */
} /* namespace ghog */

