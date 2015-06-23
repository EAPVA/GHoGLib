/*
 * HogGPU_impl.cuh
 *
 *  Created on: Jun 9, 2015
 *      Author: teider
 */

#ifndef HOGGPU_IMPL_CUH_
#define HOGGPU_IMPL_CUH_

#include <opencv2/gpu/gpu.hpp>

__global__ void gradient_kernel(float* input_img,
	float* magnitude,
	float* phase,
	int image_height,
	int image_width,
	int input_image_step,
	int magnitude_step,
	int phase_step);

__global__ void histogram_kernel(float* magnitude,
	float* phase,
	float* histograms,
	int input_width,
	int input_height,
	int magnitude_step,
	int phase_step,
	int cell_row_step,
	int cell_width,
	int cell_height,
	int num_bins);

__global__ void block_normalization_kernel(float* histograms,
	float* descriptor,
	int block_grid_width,
	int block_grid_height,
	int block_width,
	int block_height,
	int num_bins,
	int cell_grid_width,
	int block_stride_x,
	int block_stride_y);

#endif /* HOGGPU_IMPL_CUH_ */
