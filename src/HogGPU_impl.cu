#include "HogGPU_impl.cuh"

__global__ void gradient_kernel(float* input_img,
	float* magnitude,
	float* phase,
	int image_height,
	int image_width,
	int input_image_step,
	int magnitude_step,
	int phase_step)
{
	int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	if(pixel_x >= image_width)
	{
		return;
	}
	int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
	if(pixel_y >= image_height)
	{
		return;
	}

	int in_pixel_idx = pixel_y * input_image_step + pixel_x * 3;
	int mag_pixel_idx = pixel_y * magnitude_step + pixel_x;
	int phase_pixel_idx = pixel_y * phase_step + pixel_x;

	float dx, dy;
	float mag_max = 0.0f;
	float phase_max = 0.0f;

	for(int i = 0; i < 3; ++i)
	{
		dx = input_img[in_pixel_idx + 3] - input_img[in_pixel_idx - 3];
		dy = input_img[in_pixel_idx + input_image_step]
			- input_img[in_pixel_idx - input_image_step];
		float mag = sqrt(dx * dx + dy * dy);;

		if (mag > mag_max) {
			phase_max = atan2(dy, dx);
		}
	}

	magnitude[mag_pixel_idx] = mag_max;
	phase[phase_pixel_idx] = phase_max;
}

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
	int num_bins)
{
	int cell_x = blockIdx.x * blockDim.x + threadIdx.x;
	if(cell_x >= input_width)
	{
		return;
	}
	int cell_y = blockIdx.y * blockDim.y + threadIdx.y;
	if(cell_y >= input_height)
	{
		return;
	}

	int left_bin, right_bin;
	int pixel_x = cell_x * cell_width;
	int pixel_y = cell_y * cell_height;
	int mag_pixel_idx;
	int phase_pixel_idx;
	int out_idx = cell_y * cell_row_step + cell_x * num_bins;
	int i, j;

	float delta = 0.0f;
	float bin_size = 360.0f / (float)num_bins;
	float mag_total = 0;

	for(i = 0; i < cell_height; ++i)
	{
		mag_pixel_idx = (pixel_y + i) * magnitude_step + pixel_x;
		phase_pixel_idx = (pixel_y + i) * phase_step + pixel_x;
		for(j = 0; j < cell_width; ++j)
		{
			left_bin = (int)floor(
				(phase[phase_pixel_idx + j] - bin_size / 2.0f) / bin_size);
			left_bin = (left_bin + num_bins) % num_bins;
			//Might be outside the range. First use on the formula below, then fix the range.
			right_bin = (left_bin + 1);

			delta = (phase[phase_pixel_idx + j] / bin_size) - right_bin;

			//Fix range for right_bin
			right_bin = right_bin % num_bins;

			histograms[out_idx + left_bin] += (0.5 - delta)
				* magnitude[mag_pixel_idx + j];
			histograms[out_idx + right_bin] += (0.5 + delta)
				* magnitude[mag_pixel_idx + j];
			mag_total += magnitude[mag_pixel_idx + j];
		}
	}

	for(i = 0; i < num_bins; ++i)
	{
		histograms[out_idx + i] /= mag_total;
	}
}

__global__ void block_normalization_kernel(float* histograms,
	float* descriptor,
	int block_grid_width,
	int block_grid_height,
	int block_width,
	int block_height,
	int num_bins,
	int cell_grid_width,
	int block_stride_x,
	int block_stride_y)
{
	int block_x = blockIdx.x * blockDim.x + threadIdx.x;
	if(block_x >= block_grid_width)
	{
		return;
	}
	int block_y = blockIdx.y * blockDim.y + threadIdx.y;
	if(block_y >= block_grid_height)
	{
		return;
	}
	int block_idx = block_y * block_grid_width + block_x;
	int elements_per_block = block_width * block_height * num_bins;
	int block_pos = block_idx * elements_per_block;
	int block_pos_delta = 0;

	int cell_x = block_x * block_stride_x;
	int cell_y = block_y * block_stride_y;
	int cell_idx;
	int hist_pos;
	int i, j, k;

	float L1_norm = 0.0f;

	for(i = 0; i < block_height; ++i)
	{
		cell_idx = ((cell_y + i) * cell_grid_width) + cell_x;
		for(j = 0; j < block_width; ++j)
		{
			hist_pos = (cell_idx + j) * num_bins;
			for(k = 0; k < num_bins; ++k)
			{
				L1_norm += histograms[hist_pos + k];
				descriptor[block_pos + block_pos_delta] = histograms[hist_pos
					+ k];
				block_pos_delta++;
			}
		}
	}

	for(i = 0; i < elements_per_block; ++i)
	{
		descriptor[block_pos + i] /= L1_norm;
	}
}
