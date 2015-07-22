/*
 * HogCPU.h
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#ifndef GHOGLIB_HOGCPU_H_
#define GHOGLIB_HOGCPU_H_

#include <vector>
#include <string>

#include <include/HogCallbacks.h>
#include <include/IClassifier.h>
#include <include/Settings.h>

namespace ghog
{
namespace lib
{

/**
 * \brief Hog descriptor implemented on CPU.
 *
 */
class HogDescriptor
{
public:
	/**
	 * \brief XML config file constructor.
	 *
	 * Constructs a HogDescriptor based on the configurations provided by the
	 * XML config file specified. If the XML file can't be found, creates one
	 * using default parameters.
	 *
	 * \param settings_file Relative or absolute path to XML file.
	 */
	HogDescriptor(std::string settings_file);

	/**
	 * \brief Default destructor.
	 */
	virtual ~HogDescriptor();

	/**
	 * \brief Allocates a multi-channel 2D buffer.
	 *
	 * Not really relevant to the CPU implementation. Preallocates a buffer to
	 * be used inside the library.
	 *
	 * \param buffer_size Dimensions of the desired buffer.
	 * \param type Specification of color depth and number of channels of the
	 * buffer. Uses the system implemented in OpenCV.
	 * \param padding_size Specifies a zero padding border, in all directions,
	 * for the buffer The returned buffer doesn't contain the border (the border
	 * is outside of the buffer).
	 *
	 * \see HogGPU::alloc_buffer(cv::Size buffer_size, int type,
	 * cv::Mat& buffer,	int padding_size)
	 */
	virtual GHOG_LIB_STATUS alloc_buffer(cv::Size buffer_size,
		int type,
		cv::Mat& buffer,
		int padding_size);

	/**
	 * \brief Asynchronous image normalization.
	 *
	 * Calls image_normalization_sync(cv::Mat& image) in another thread.
	 *
	 * \param image Input image.
	 * \param callback Callback object to be called when the function finishes.
	 */
	virtual GHOG_LIB_STATUS image_normalization(cv::Mat& image,
		ImageCallback* callback);
	/**
	 * \brief Synchronous image normalization.
	 *
	 * Normalizes input image to the [0,1) interval (without contrast
	 * normalization, just dividing by 256) and also performs square root
	 * normalization. The function expects an image of type CV_32FC3 (3 channels
	 * with 32-bit color depth).
	 *
	 * \todo Add support to other input image types.
	 *
	 * \param image Input image.
	 */
	virtual GHOG_LIB_STATUS image_normalization_sync(cv::Mat& image);

	/**
	 * \brief Asynchronous gradient calculation.
	 *
	 * Calls calc_gradient_sync(cv::Mat input_img,cv::Mat& magnitude,cv::Mat& phase)
	 * in another thread.
	 *
	 * \param image Input image.
	 * \param magnitude 2D buffer to place the magnitudes of the gradients.
	 * \param phase 2D buffer to place the orientations (phase) of the gradients.
	 * \param callback Callback object to be called when the function finishes.
	 */
	virtual GHOG_LIB_STATUS calc_gradient(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase,
		GradientCallback* callback);
	/**
	 * \brief Synchronous gradient calculation.
	 *
	 * Calculates gradients orientations and magnitudes for the whole image. For
	 * each pixel, the gradient with the largest magnitude is chosen. The
	 * function expects an input image of type CV_32FC3 (3 channels
	 * with 32-bit color depth) and output buffers with type CV_32FC1 (1 channel
	 * with 32-bit color depth). The input image must be at least one pixel
	 * wider in all directions than the cv::Mat that is passed to the function.
	 *
	 * \todo Verify size and type of the buffers.
	 * \todo Verify that the input image has a border.
	 * \todo Add support to grayscale images.
	 *
	 * \param image Input image.
	 * \param magnitude 2D buffer to place the magnitudes of the gradients.
	 * \param phase 2D buffer to place the orientations (phase) of the gradients.
	 */
	virtual GHOG_LIB_STATUS calc_gradient_sync(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase);

	/**
	 * \brief Asynchronous descriptor calculation.
	 *
	 * Calls create_descriptor_sync(cv::Mat magnitude,cv::Mat phase,cv::Mat& descriptor)
	 * in another thread.
	 *
	 * \todo Add interface to obtain histograms as well.
	 *
	 * \param magnitude 2D buffer with the magnitudes of the gradients.
	 * \param phase 2D buffer with the orientations (phase) of the gradients.
	 * \param descriptor Buffer to place the resulting descriptor.
	 * \param callback Callback object to be called when the function finishes.
	 */
	virtual GHOG_LIB_STATUS create_descriptor(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor,
		DescriptorCallback* callback);
	/**
	 * \brief Synchronous descriptor calculation.
	 *
	 * Allocates a histogram buffer and calls
	 * create_descriptor_sync(cv::Mat magnitude,cv::Mat phase,cv::Mat& descriptor,cv::Mat& histograms)
	 *
	 * \param magnitude 2D buffer with the magnitudes of the gradients.
	 * \param phase 2D buffer with the orientations (phase) of the gradients.
	 * \param descriptor Buffer to place the resulting descriptor.
	 */
	virtual GHOG_LIB_STATUS create_descriptor_sync(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor);
	/**
	 * \brief Synchronous descriptor calculation.
	 *
	 * Calculates each histogram of each cell, copies each histogram to the
	 * corresponding position on the descriptor buffer and then normalizes
	 * the buffer. Both magnitude and phase buffers must have the same size and
	 * it must match the size of the detection window (size of each cell *
	 * size of the cell grid). The descriptor must be a line vector with the
	 * number of elements equal to the dimension of the resulting descriptor
	 * (as returned by the function get_descriptor_size() ).
	 *
	 * \param magnitude 2D buffer with the magnitudes of the gradients.
	 * \param phase 2D buffer with the orientations (phase) of the gradients.
	 * \param descriptor Buffer to place the resulting descriptor.
	 * \param histograms Buffer with the histograms of each cell.
	 */
	virtual GHOG_LIB_STATUS create_descriptor_sync(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor,
		cv::Mat& histograms);

	/**
	 * \brief Asynchronous classification.
	 *
	 * Calls classify_sync(cv::Mat img) in another thread.
	 *
	 * \param img The image to be classified.
	 * \param callback Callback object to be called after the function finishes.
	 */
	virtual GHOG_LIB_STATUS classify(cv::Mat img,
		ClassifyCallback* callback);
	/**
	 * \brief Synchronous classification.
	 *
	 * Performs all three steps (normalization, gradient calculation and
	 * descriptor calculation on the input image and then classifies it using
	 * a classifier previously provided by the function
	 * set_classifier(IClassifier* classifier).
	 *
	 * \todo Verify if the classifier was set.
	 *
	 * \param img The image to be classified.
	 */
	virtual bool classify_sync(cv::Mat img);

	/**
	 * \brief Asynchronous detection
	 *
	 * \warning Not implemented yet.
	 *
	 * Calls locate_sync(cv::Mat img,cv::Rect roi,cv::Size window_size,cv::Size window_stride)
	 * in another thread.
	 *
	 * \param img Input image.
	 * \param roi Region Of Interest to be searched on the input image. Pixels
	 * outside the ROI are used on the gradient calculation.
	 * \param window_size Detection window dimensions.
	 * \param window_stride Detection window stride.
	 * \param callback Callback object to be called after the function finishes.
	 */
	virtual GHOG_LIB_STATUS locate(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride,
		LocateCallback* callback);
	/**
	 * \brief Synchronous detection
	 *
	 * \warning Not implemented yet.
	 *
	 * Performs single-scale object detection on the input image, using a
	 * sliding window algorithm. Uses the classifier previously provided by the
	 * function set_classifier(IClassifier* classifier).
	 *
	 * \param img Input image.
	 * \param roi Region Of Interest to be searched on the input image. Pixels
	 * outside the ROI are used on the gradient calculation.
	 * \param window_size Detection window dimensions.
	 * \param window_stride Detection window stride.
	 * \param callback Callback object to be called after the function finishes.
	 */
	virtual std::vector< cv::Rect > locate_sync(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride);

	/**
	 * \brief Loads settings from XML file.
	 *
	 * Sets HOG parameters based on the configurations found on a XML file.
	 * If the file can't be found, creates one using default parameters.
	 *
	 * \param filename Relative or absolute path to XML file.
	 */
	void load_settings(std::string filename);

	/**
	 * \brief Sets the classifier.
	 *
	 * The classifier specified is used by the functions
	 * classify_sync(cv::Mat img) and
	 * locate_sync(cv::Mat img,cv::Rect roi,cv::Size window_size,cv::Size window_stride).
	 *
	 * \param classifier The classifier to be used.
	 */
	void set_classifier(IClassifier* classifier);

	/**
	 * \brief Changes a HOG configuration parameter.
	 *
	 * \bug Does not save the value to the file. As the value is then read from
	 * the file, it doesn't change the parameter.
	 *
	 * \todo Add configuration for contrast normalization.
	 *
	 * \param param Name of the parameter to be changed.
	 * \param value New value of the parameter.
	 */
	GHOG_LIB_STATUS set_param(std::string param,
		std::string value);
	/**
	 * \brief Reads a HOG configuration parameter.
	 *
	 * \param param Name of the paramter to be read.
	 *
	 * \todo Add configuration for contrast normalization.
	 *
	 * \returns Value of the parameter.
	 */
	std::string get_param(std::string param);

	/**
	 * \brief Returns the size of the resulting descriptor, based on current
	 * configuration parameters.
	 */
	int get_descriptor_size();

protected:
	/**
	 * \brief Helper function for asynchronous operation.
	 *
	 * \see image_normalization(cv::Mat& image,ImageCallback* callback)
	 */
	void image_normalization_async(cv::Mat& image,
		ImageCallback* callback);
	/**
	 * \brief Helper function for asynchronous operation
	 *
	 * \see calc_gradient(cv::Mat input_img,cv::Mat& magnitude,cv::Mat& phase,GradientCallback* callback)
	 */
	void calc_gradient_async(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase,
		GradientCallback* callback);
	/**
	 * \brief Helper function for asynchronous operation.
	 *
	 * \see create_descriptor(cv::Mat magnitude,cv::Mat phase,cv::Mat& descriptor,DescriptorCallback* callback)
	 */
	void create_descriptor_async(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor,
		DescriptorCallback* callback);

	/**
	 * \brief Helper function for asynchronous operation.
	 *
	 * \see classify(cv::Mat img,ClassifyCallback* callback)
	 */
	void classify_async(cv::Mat img,
		ClassifyCallback* callback);

	/**
	 * \brief Helper function for asynchronous operation.
	 *
	 * \see locate(cv::Mat img,cv::Rect roi,cv::Size window_size,cv::Size window_stride,LocateCallback* callback)
	 */
	void locate_async(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride,
		LocateCallback* callback);

	/**
	 * \brief Calculates the histogram of one cell.
	 *
	 * The histogram is placed on a n-channel matrix with one element, with n
	 * being the number of classes in the histogram.
	 *
	 * \param magnitude Magnitudes of the gradients of the cell.
	 * \param phase Orientations of the gradients of the cell.
	 * \param cell_histogram Output histogram
	 */
	void calc_histogram(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat cell_histogram);

	/**
	 * \brief Performs local normalization on the descriptor.
	 *
	 * The descriptor is partitioned into blocks, and each block is normalized
	 * using the contrast normalization specified in the configuration.
	 * Currently supports L1-sqrt and L2-Hys.
	 *
	 * \param descriptor The descriptor to be normalized.
	 */
	void normalize_blocks(cv::Mat& descriptor);

	/**
	 * \brief Returns the name of the module of the configuration parameter,
	 * based on the name of the parameter.
	 *
	 * \param param_name Name of the parameter.
	 *
	 * \returns Name of the module.
	 */
	std::string get_module(std::string param_name);

	Settings _settings; /**<Responsible for XML read/write.*/

	IClassifier* _classifier; /**<Classifier used internally.*/

	int _num_bins; /**<Number of classes on each histograms.*/
	cv::Size _cell_size; /**<Cell dimensions, in number of pixels / cell.*/
	cv::Size _block_size; /**<Block dimensions, in number of cells / block.*/
	cv::Size _block_stride; /**<Block stride, in number of cells.
	 (How many cells to move the block window after calculating one)*/
	cv::Size _cell_grid; /**<Size of the detection window, in number of cells.*/
	cv::Size _window_size; /**<Size of the detection window, in number of
	 pixels, calculated based on other parameters*/

	GHOG_LIB_NORM_TYPE _norm_type; /**<Type of normalization used.
	 \see #GHOG_LIB_NORM_TYPE */
};

} /* namespace lib */
} /* namespace ghog */

#endif /* GHOGLIB_HOGCPU_H_ */
