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

	int in_pixel_idx = pixel_y * input_image_step + pixel_x;
	int mag_pixel_idx = pixel_y * magnitude_step + pixel_x;
	int phase_pixel_idx = pixel_y * phase_step + pixel_x;

	float dx = input_img[in_pixel_idx + 1] - input_img[in_pixel_idx - 1];
	float dy = input_img[in_pixel_idx + input_image_step]
		- input_img[in_pixel_idx - input_image_step];

	magnitude[mag_pixel_idx] = sqrt(dx * dx + dy * dy);
	phase[phase_pixel_idx] = atan2(dy, dx);
}
