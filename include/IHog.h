/*
 * IHog.h
 *
 *  Created on: Jun 1, 2015
 *      Author: marcelo
 */

#ifndef IHOG_H_
#define IHOG_H_

#include <vector>

#include <opencv2/core/core.hpp>

#include <include/GHogLibConstants.inc>
#include <include/HogCallbacks.h>

namespace ghog
{
namespace lib
{

class IHog
{
public:
	virtual ~IHog()
	{
	}
	;

	virtual void alloc_buffer(cv::Size buffer_size,
		int type,
		cv::Mat& buffer) = 0;

	virtual GHOG_LIB_STATUS image_normalization(cv::Mat& image,
		ImageCallback* callback) = 0;

	virtual void image_normalization_sync(cv::Mat& image) = 0;

	virtual GHOG_LIB_STATUS calc_gradient(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase,
		GradientCallback* callback) = 0;

	virtual void calc_gradient_sync(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase) = 0;

	virtual GHOG_LIB_STATUS create_descriptor(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor,
		DescriptorCallback* callback) = 0;

	virtual void create_descriptor_sync(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor) = 0;

	virtual GHOG_LIB_STATUS classify(cv::Mat img,
		ClassifyCallback* callback) = 0;

	virtual bool classify_sync(cv::Mat img) = 0;

	virtual GHOG_LIB_STATUS locate(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride,
		LocateCallback* callback) = 0;

	virtual std::vector< cv::Rect > locate_sync(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride) = 0;

	virtual void load_settings(std::string filename) = 0;

	virtual void set_classifier(IClassifier* classifier) = 0;

	virtual GHOG_LIB_STATUS set_param(std::string param,
		std::string value) = 0;
	virtual std::string get_param(std::string param) = 0;
};

} /* namespace lib */
} /* namespace ghog */

#endif /* IHOG_H_ */
