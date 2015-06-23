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

class HogGPU: public HogDescriptor
{
public:
	HogGPU(std::string settings_file);
	virtual ~HogGPU();

	void alloc_buffer(cv::Size buffer_size,
		int type,
		cv::Mat& buffer);

	GHOG_LIB_STATUS image_normalization(cv::Mat& image,
		ImageCallback* callback);
	void image_normalization_sync(cv::Mat& image);

	GHOG_LIB_STATUS calc_gradient(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase,
		GradientCallback* callback);
	void calc_gradient_sync(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase);

	virtual GHOG_LIB_STATUS create_descriptor(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor,
		DescriptorCallback* callback);
	virtual void create_descriptor_sync(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor);
};

} /* namespace lib */
} /* namespace ghog */

#endif /* HOGGPU_H_ */
