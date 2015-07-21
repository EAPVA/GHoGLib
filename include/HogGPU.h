/*
 * HogGPU.h
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#ifndef HOGGPU_H_
#define HOGGPU_H_

#include <vector>
#include <string>

#include <include/IClassifier.h>
#include <include/Settings.h>
#include <include/HogDescriptor.h>

namespace ghog
{
namespace lib
{

/**
 * \brief Hog descriptor implemented on GPU.
 *
 * The GPU kernel functions are defined in HogGPU_impl.cuh
 *
 */
class HogGPU: public HogDescriptor
{
public:
	/**
	 * \brief XML config file constructor.
	 *
	 * Constructs a HogDescriptor based on the configurations provided by the
	 * XML config file provided. If the XML file can't be found, creates one
	 * using default parameters.
	 *
	 * \param settings_file Relative or absolute path to XML file.
	 */
	HogGPU(std::string settings_file);
	/**
	 * \brief Default destructor.
	 */
	virtual ~HogGPU();

	/**
	 * \brief Allocates a multi-channel 2D buffer.
	 *
	 * The buffer is allocated as a portion of shared memory between
	 * the CPU and the GPU, which allows to use Zero Copy Memory.
	 * In order to align GPU data access, each row is
	 * stored with a size multiple of a parameter that depends on hardware (For
	 * Tegra K1 is 512 bytes). Because of that, matrices allocated aren't
	 * guaranteed to be continuous (i.e.: the first element of a row is stored
	 * right after the last element of the previous row).
	 *
	 * \param buffer_size Dimensions of the desired buffer.
	 * \param type Specification of color depth and number of channels of the
	 * buffer. Uses the system implemented in OpenCV.
	 * \param padding_size Specifies a zero padding border, in all directions,
	 * for the buffer The returned buffer doesn't contain the border (the border
	 * is outside of the buffer).
	 */
	GHOG_LIB_STATUS alloc_buffer(cv::Size buffer_size,
		int type,
		cv::Mat& buffer,
		int padding_size);

	GHOG_LIB_STATUS image_normalization(cv::Mat& image,
		ImageCallback* callback);
	GHOG_LIB_STATUS image_normalization_sync(cv::Mat& image);

	GHOG_LIB_STATUS calc_gradient(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase,
		GradientCallback* callback);
	GHOG_LIB_STATUS calc_gradient_sync(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase);

	virtual GHOG_LIB_STATUS create_descriptor(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor,
		DescriptorCallback* callback);
//	virtual void create_descriptor_sync(cv::Mat magnitude,
//		cv::Mat phase,
//		cv::Mat& descriptor);
	virtual GHOG_LIB_STATUS create_descriptor_sync(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor,
		cv::Mat& histograms);
};

} /* namespace lib */
} /* namespace ghog */

#endif /* HOGGPU_H_ */
