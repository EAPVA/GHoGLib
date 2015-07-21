#include "HogGPU_impl.cuh"

#include "math_constants.h"

namespace ghog
{
namespace lib
{
namespace gpu
{

__global__ void gamma_norm_kernel(float* img,
	int image_height,
	int image_width,
	int image_step)
{
	// The thread block has size (3,n). The first dimension of the thread block
	// corresponds to color channels.
	int channel = threadIdx.x;
	// The columns of the image are mapped to the first dimension of the block
	// grid, but to the second dimension of the thread block, as the first
	// already corresponds to color channels.
	int pixel_x = blockIdx.x * blockDim.y + threadIdx.y;
	// If current position is outside the image, stop here
	if(pixel_x >= image_width)
	{
		return;
	}
	// The columns of the image are mapped to the second dimension of the block
	// grid, but to the third dimension of the thread block.
	int pixel_y = blockIdx.y * blockDim.z + threadIdx.z;
	// If current position is outside the image, stop here
	if(pixel_y >= image_height)
	{
		return;
	}

	// Each row has image_step pixels and each pixel has three channels
	int in_pixel_idx = pixel_y * image_step + pixel_x * 3 + channel;

	// Finally perform the normalization
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
	//TODO: make the buffer sizes dependent on an input or template parameter.
	//Each thread block needs to store 2 * 64 bytes * 3 floats per channel = 2 * 192
	__shared__ float s_magnitude[192];
	__shared__ float s_phase[192];

	// The thread block has size (3,n). The first dimension of the thread block
	// corresponds to color channels.
	int channel = threadIdx.x;
	// The columns of the image are mapped to the first dimension of the block
	// grid, but to the second dimension of the thread block, as the first
	// already corresponds to color channels.
	int pixel_x = blockIdx.x * blockDim.y + threadIdx.y;
	// If current position is outside the image, stop here
	if(pixel_x >= image_width)
	{
		return;
	}
	// The columns of the image are mapped to the second dimension of the block
	// grid, but to the third dimension of the thread block.
	int pixel_y = blockIdx.y * blockDim.z + threadIdx.z;
	// If current position is outside the image, stop here
	if(pixel_y >= image_height)
	{
		return;
	}

	//The indexes for the internal buffer don't depend on the block index.
	int bs_x = threadIdx.y;
	int bs_y = threadIdx.z;
	int bs_step = 3 * blockDim.y;
	int bs_idx = bs_y * bs_step + bs_x * 3 + channel;

	// Each row has input_image_step size and each pixel has three channels
	int in_pixel_idx = pixel_y * input_image_step + pixel_x * 3 + channel;
	// Each row has magnitude_step size
	int mag_pixel_idx = pixel_y * magnitude_step + pixel_x;
	// Each row has phase_step size
	int phase_pixel_idx = pixel_y * phase_step + pixel_x;

	// Calculate the X and Y coordinates of the gradient.
	float dx = input_img[in_pixel_idx + 3];
	dx -= input_img[in_pixel_idx - 3];
	float dy = input_img[in_pixel_idx + input_image_step];
	dy -= input_img[in_pixel_idx - input_image_step];

	// Store the magnitude and the phase of the gradient on the shared buffer.
	s_magnitude[bs_idx] = sqrt(dx * dx + dy * dy);
	// Normalize the phase output to [0,1] rotations.
	s_phase[bs_idx] = (atan2(dy, dx) + CUDART_PI_F) / (2.0f * CUDART_PI_F);

	// Wait until all threads finish this step.
	__syncthreads();

	//Only one each three threads will verify the max value and store the result.
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
			mag_max = s_magnitude[3 * threadIdx.y + 2];
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
	//TODO: make the buffer sizes dependent on an input or template parameter.
	// Each thread block needs to store intermediate results for 64 gradients
	// and also 8 different histograms, each with 9 bins.
	__shared__ int s_lbin_pos[64];
	__shared__ float s_lbin[64];
	__shared__ int s_rbin_pos[64];
	__shared__ float s_rbin[64];
	__shared__ float s_hist[9 * 8];

	// The columns of the image are mapped to the first dimension of the block
	// grid and the first dimension of the thread block.
	int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	// If current position is outside the image, stop here
	if(pixel_x >= input_width)
	{
		return;
	}
	// The columns of the image are mapped to the second dimension of the block
	// grid and the second dimension of the thread block.
	int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
	// If current position is outside the image, stop here
	if(pixel_y >= input_height)
	{
		return;
	}

	// Each row has magnitude_step size
	int mag_pixel_idx = pixel_y * magnitude_step + pixel_x;
	// Each row has phase_step size
	int phase_pixel_idx = pixel_y * phase_step + pixel_x;

	// The phase was previously normalized to [0,1]
	float bin_size = 1.0f / (float)num_bins;
	// By dividing by the bin size and taking the integer part, you find out
	// inside which bin the gradient is at. If it's greater than the middle of the bin
	// it will be divided between this one and the next, if it's lesser it will
	// be divided between this and the previous one. By subtracting 0.5 before
	// taking the integer part, the division will always be between this bin and
	// the next.
	int left_bin = (int)floor((phase[phase_pixel_idx] / bin_size) - 0.5f);
	// The result of the previous operation might be negative. If so, the next
	// bit fixes that. Otherwise that changes nothing.
	left_bin = (left_bin + num_bins) % num_bins;
	// Take the next bin as the right bin.
	// If the left bin is the last one, this will be outside range. Wait a bit
	// before taking the remainder, because this value needs to be used in the
	// formula below.
	int right_bin = (left_bin + 1);
	// Calculate the distance between the gradient phase and the limit between
	// the left and right bins. Normalized by the bin size, the limit is equal
	// to the right bin identifier.
	float delta = (phase[phase_pixel_idx] / bin_size) - right_bin;
	if(delta < -0.5)
	{
		delta += num_bins;
	}
	//Fix range for right_bin now
	right_bin = right_bin % num_bins;

	// Store the bin positions and amounts for each bin on shared buffers.
	s_lbin_pos[threadIdx.x] = left_bin;
	s_lbin[threadIdx.x] = (0.5 - delta) * magnitude[mag_pixel_idx];
	s_rbin_pos[threadIdx.x] = right_bin;
	s_rbin[threadIdx.x] = (0.5 + delta) * magnitude[mag_pixel_idx];

	// Wait for other threads.
	__syncthreads();

	// Initialize histograms shared buffer.
	s_hist[threadIdx.x] = 0.0f;
	if(threadIdx.x < 8)
	{
		s_hist[threadIdx.x + 64] = 0.0f;
	}

	int cell_y = pixel_y / cell_height;

	// Each partial histogram will be calculated by only one thread.
	if(threadIdx.x < 8)
	{
		int s_hist_idx = 9 * threadIdx.x;
		for(int i = 1; i < 8; ++i)
		{
			s_hist[s_hist_idx + s_lbin_pos[8 * threadIdx.x + i]] += s_lbin[8
				* threadIdx.x + i];
			s_hist[s_hist_idx + s_rbin_pos[8 * threadIdx.x + i]] += s_rbin[8
				* threadIdx.x + i];
		}
	}

	// Wait until all threads finish.
	__syncthreads();

	// Add to the complete histogram sum using atomic operations.
	int out_idx = cell_y * histograms_step + threadIdx.x;
	atomicAdd(&(histograms[out_idx]), s_hist[threadIdx.x]);

	if(threadIdx.x < 8)
	{
		atomicAdd(&(histograms[out_idx + 64]), s_hist[threadIdx.x + 64]);
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
	//TODO: make the buffer sizes dependent on an input or template parameter.
	// Each thread block will process 8 hog blocks. Each hog block has 4 cells.
	// Each cell has 9 bins.
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
		s_blocks[s_blocks_idx] / L1_norm[threadIdx.z];
}

} /* namespace gpu */
} /* namespace ghog */
} /* namespace lib */
