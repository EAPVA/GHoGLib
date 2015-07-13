#include "HogGPU_impl.cuh"

#include "math_constants.h"

__global__ void gamma_norm_kernel(float* img,
	int image_height,
	int image_width,
	int image_step)
{
	int channel = threadIdx.x;
	int pixel_x = blockIdx.x * blockDim.y + threadIdx.y;
	if(pixel_x >= image_width)
	{
		return;
	}
	int pixel_y = blockIdx.y * blockDim.z + threadIdx.z;
	if(pixel_y >= image_height)
	{
		return;
	}

	int in_pixel_idx = pixel_y * image_step + pixel_x * 3 + channel;

	img[in_pixel_idx] = sqrt(img[in_pixel_idx] / 256.0f);

}

__global__ void gradient_kernel(float* input_img,
	float* magnitude,
	float* phase,
	int image_height,
	int image_width,
	int input_image_step,
	int magnitude_step,
	int phase_step)
{
	__shared__ float s_magnitude[192];
	__shared__ float s_phase[192];

	int channel = threadIdx.x;
	int pixel_x = blockIdx.x * blockDim.y + threadIdx.y;
	if(pixel_x >= image_width)
	{
		return;
	}
	int pixel_y = blockIdx.y * blockDim.z + threadIdx.z;
	if(pixel_y >= image_height)
	{
		return;
	}

	int bs_x = threadIdx.y;
	int bs_y = threadIdx.z;
	int bs_step = 3 * blockDim.y;
	int bs_idx = bs_y * bs_step + bs_x * 3 + channel;

	int in_pixel_idx = pixel_y * input_image_step + pixel_x * 3 + channel;
	int mag_pixel_idx = pixel_y * magnitude_step + pixel_x;
	int phase_pixel_idx = pixel_y * phase_step + pixel_x;

	float dx = input_img[in_pixel_idx + 3];
	dx -= input_img[in_pixel_idx - 3];
	float dy = input_img[in_pixel_idx + input_image_step];
	dy -= input_img[in_pixel_idx - input_image_step];

	s_magnitude[bs_idx] = sqrt(dx * dx + dy * dy);
	s_phase[bs_idx] = (atan2(dy, dx) + CUDART_PI_F) / (2.0f * CUDART_PI_F);

	__syncthreads();

	if(channel == 0)
	{
		float mag_max = s_magnitude[3 * threadIdx.y];
		int k = 0;
		if(s_magnitude[3 * threadIdx.y + 1] > mag_max)
		{
			mag_max = s_magnitude[3 * threadIdx.y + 1];
			k = 1;
		}
		if(s_magnitude[3 * threadIdx.y + 2] > mag_max)
		{
			mag_max = s_magnitude[3 * threadIdx.y + 1];
			k = 2;
		}

		magnitude[mag_pixel_idx] = mag_max;
		phase[phase_pixel_idx] = s_phase[3 * threadIdx.y + k];
	}
}

__global__ void histogram_kernel(float* magnitude,
	float* phase,
	float* histograms,
	int input_width,
	int input_height,
	int cell_grid_width,
	int cell_grid_height,
	int magnitude_step,
	int phase_step,
	int histograms_step,
	int cell_width,
	int cell_height,
	int num_bins)
{
	__shared__ int s_lbin_pos[64];
	__shared__ float s_lbin[64];
	__shared__ int s_rbin_pos[64];
	__shared__ float s_rbin[64];
	__shared__ float s_hist[9 * 2];
	__shared__ float s_hist_total[2];

	if(threadIdx.x < 18)
	{
		s_hist[threadIdx.x] = 0.0f;
	}
	if(threadIdx.x < 2)
	{
		s_hist_total[threadIdx.x] = 0.0f;
	}

	int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	if(pixel_x >= input_width)
	{
		return;
	}
	int pixel_y = 32 * (blockIdx.y * blockDim.y + threadIdx.y);
	if(pixel_y >= input_height)
	{
		return;
	}
	int cell_y = pixel_y / cell_height;
	int cell_x = pixel_x / cell_width;

	for(int i = 0; i < 32; ++i)
	{
		int mag_pixel_idx = pixel_y * magnitude_step + pixel_x;
		int phase_pixel_idx = pixel_y * phase_step + pixel_x;

		float bin_size = 1.0f / (float)num_bins;
		int left_bin = (int)floor(
			(phase[phase_pixel_idx] - bin_size / 2.0f) / bin_size);
		left_bin = (left_bin + num_bins) % num_bins;
		//Might be outside the range. First use on the formula below, then fix the range.
		int right_bin = (left_bin + 1);
		float delta = (phase[phase_pixel_idx] / bin_size) - right_bin;
		if(delta < -0.5)
		{
			delta += num_bins;
		}
		//Fix range for right_bin
		right_bin = right_bin % num_bins;

		s_lbin_pos[threadIdx.x] = left_bin;
		s_lbin[threadIdx.x] = (0.5 - delta) * magnitude[mag_pixel_idx];
		s_rbin_pos[threadIdx.x] = right_bin;
		s_lbin[threadIdx.x] = (0.5 + delta) * magnitude[mag_pixel_idx];

//	s_hist[threadIdx.x] = 0.0f;

		__syncthreads();

		if(threadIdx.x < 2)
		{
			int s_hist_idx = 9 * threadIdx.x;
			for(int i = 0; i < 32; ++i)
			{
				s_hist[s_hist_idx + s_lbin_pos[32 * threadIdx.x + i]] +=
					s_lbin[32 * threadIdx.x + i];
				s_hist[s_hist_idx + s_rbin_pos[32 * threadIdx.x + i]] +=
					s_rbin[32 * threadIdx.x + i];
				s_hist_total[threadIdx.x] += s_lbin[32 * threadIdx.x + i]
					+ s_rbin[32 * threadIdx.x + i];
			}
		}
		pixel_y++;

		__syncthreads();
	}

	int cell_pos = threadIdx.x / 9;
	int out_idx = histograms_step * cell_y + num_bins * (cell_x) + threadIdx.x;
//	return;

	if(threadIdx.x < 18)
	{
		if(s_hist_total[cell_pos] > 0.1)
		{
			s_hist[threadIdx.x] /= s_hist_total[cell_pos];
		}
		histograms[out_idx] = s_hist[threadIdx.x];
	}
}

__global__ void block_normalization_kernel(float* histograms,
	float* descriptor,
	int histograms_step,
	int block_grid_width,
	int block_grid_height,
	int block_width,
	int block_height,
	int num_bins,
	int cell_grid_width,
	int block_stride_x,
	int block_stride_y)
{
	//Each thread block will process 8 hog blocks.
	__shared__ float s_blocks[9 * 4 * 8];
	__shared__ float L1_norm[8];
	int block_x = blockIdx.x * 8 + threadIdx.z;
	if(block_x >= block_grid_width)
	{
		return;
	}
	int block_y = blockIdx.y;
	if(block_y >= block_grid_height)
	{
		return;
	}
	int block_idx = block_y * blockDim.y + block_x;
	int cell_x = block_x * block_stride_x + threadIdx.y % 2;
	int cell_y = block_y * block_stride_y + threadIdx.y / 2;
	int hist_idx = histograms_step * cell_y + num_bins * (cell_x) + threadIdx.x;

	int s_blocks_idx = 9 * threadIdx.y + threadIdx.x;
	s_blocks[s_blocks_idx] = histograms[hist_idx];

	__syncthreads();

	int thread_id = 36 * threadIdx.z + 9 * threadIdx.y + threadIdx.x;
	int elements_per_block = block_height * block_width * num_bins;
	if(thread_id < 8)
	{
		L1_norm[thread_id] = 0.0f;
		for(int i = 0; i < elements_per_block; ++i)
		{
			L1_norm[thread_id] += s_blocks[elements_per_block * thread_id + i];
		}
	}

	__syncthreads();

	descriptor[elements_per_block * block_idx + s_blocks_idx] =
		s_blocks[s_blocks_idx];
}
