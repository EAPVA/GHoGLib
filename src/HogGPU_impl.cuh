/*
 * HogGPU_impl.cuh
 *
 *  Created on: Jun 9, 2015
 *      Author: teider
 */

#ifndef HOGGPU_IMPL_CUH_
#define HOGGPU_IMPL_CUH_

namespace ghog
{
namespace lib
{
namespace gpu
{

/** \file
 *
 * \brief Kernel functions used in the GPU implementation.
 *
 * \todo Generalize functions to other thread block sizes.
 *
 */

/**
 * \brief Image normalization.
 *
 * Each thread calculates the normalization of one color channel of
 * one pixel of the image.
 *
 * \param img Pointer to first element of input image.
 * \param image_height Image height, used if the image size isn't
 * divisible by the threads block size, to avoid accessing elements outside the
 * image.
 * \param image_width Image width, used if the image size isn't
 * divisible by the thread block size, to avoid accessing elements outside the
 * image.
 * \param image_step Size of the image row (including the extra bytes needed
 * for GPU data alignment).
 *
 */
__global__ void gamma_norm_kernel(float* img,
	int image_height,
	int image_width,
	int image_step);

/**
 * \brief Gradient calculation
 *
 * Each thread calculates the magnitude and orientation of the gradient of one
 * color channel of one pixel of the input image. After that, one each three
 * threads finds the one with the maximum magnitude, from the three of the
 * corresponding pixel, and stores both magnitude and phase on the output
 * buffers.
 *
 * \param input_img Pointer to the first element of the input image.
 * \param magnitude Pointer to the first element of the magnitude output buffer.
 * \param phase Pointer to the first element of the phase output buffer.
 * \param image_height Image height, used if the image size isn't
 * divisible by the threads block size, to avoid accessing elements outside the
 * image.
 * \param image_width Image width, used if the image size isn't
 * divisible by the threads block size, to avoid accessing elements outside the
 * image.
 * \param input_image_step Size of the image row (including the extra bytes
 * needed for GPU data alignment).
 * \param magnitude_step Size of the magnitude buffer row (including the extra
 * bytes needed for GPU data alignment).
 * \param phase_step Size of the phase buffer row (including the extra bytes
 * needed for GPU data alignment).
 */
__global__ void gradient_kernel(float* input_img,
	float* magnitude,
	float* phase,
	int image_height,
	int image_width,
	int input_image_step,
	int magnitude_step,
	int phase_step);

/**
 * \brief Calculates the orientation histograms.
 *
 * Each thread block calculates a partial sum of several cells, taking a subrow
 * from the buffers and calculating the partial result on the corresponding
 * histograms. On each thread block, the process is done using three steps.
 * First, each thread reads the magnitude and orientation of one gradient and
 * calculates which histogram classes will be updated by that gradient, and how
 * much will be added to each one, storing this information on shared memory.
 * Then a small number of threads calculates the partial histogram of each cell,
 * placing the result on shared memory. Finally the histograms buffer with the
 * partial sums is divided between each thread of the block to atomic add to the
 * complete histogram sum.
 *
 * \param magnitude Pointer to the first element of the gradient magnitude buffer.
 * \param phase Pointer to the first element of the gradient phase buffer.
 * \param histograms Pointer to the first element of the histogram buffer.
 * \param image_width Image width, used if the image size isn't
 * divisible by the threads block size, to avoid accessing elements outside the
 * image.
 * \param image_height Image height, used if the image size isn't
 * divisible by the threads block size, to avoid accessing elements outside the
 * image.
 * \param cell_grid_width Number of cells horizontally on the cell grid.
 * \param cell_grid_height Number of cells vertically on the cell grid.
 * \param magnitude_step Size of the magnitude buffer row (including the extra
 * bytes needed for GPU data alignment).
 * \param phase_step Size of the phase buffer row (including the extra bytes
 * needed for GPU data alignment).
 * \param histograms_step Size of the histograms buffer row (including the extra
 * bytes needed for GPU data alignment).
 * \param cell_width Width of cell, in pixels.
 * \param cell_height Height of cell, in pixels.
 * \param num_bins Number of cells in the histogram.
 */
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
	int num_bins);

/**
 * \brief Performs HOG's block normalization
 *
 * Each block of threads normalizes several hog blocks at once. The
 * normalization is processed in three steps: First each thread loads the input
 * to shared memory, then a smaller number of threads calculates the norm and
 * finally each thread normalizes the element it had loaded on the first step.
 *
 * \param histograms Pointer to the first element of the histograms buffer.
 * \param descriptor Pointer to the first element of the descriptor buffer.
 * \param histograms_step Size of the histograms buffer row (including the extra
 * bytes needed for GPU data alignment).
 * \param block_grid_width Number of blocks horizontally on the detection window.
 * \param block_grid_height Number of blocks vertically on the detection window.
 * \param magnitude_step Size of the magnitude buffer row (including the extra
 * bytes needed for GPU data alignment).
 * \param block_width Width of block, in number of cells.
 * \param block_height Height of block, in number of cells.
 * \param num_bins Number of classes in the histogram.
 * \param cell_grid_width Number of cells horizontally on the cell grid.
 * \param block_stride_x Horizontal stride of block, in number of cells.
 * \param block_stride_y Vertical stride of block, in number of cells.
 */
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
	int block_stride_y);

} /* namespace gpu */
} /* namespace ghog */
} /* namespace lib */

#endif /* HOGGPU_IMPL_CUH_ */
