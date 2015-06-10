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

#endif /* HOGGPU_IMPL_CUH_ */
