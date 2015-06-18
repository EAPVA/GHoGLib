/*
 * IHog.h
 *
 *  Created on: Jun 1, 2015
 *      Author: marcelo
 */

#ifndef IHOG_H_
#define IHOG_H_

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

	virtual GHOG_LIB_STATUS resize(cv::Mat image,
		cv::Size new_size,
		cv::Mat& resized_image,
		ImageCallback* callback) = 0;

	virtual GHOG_LIB_STATUS calc_gradient(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase,
		GradientCallback* callback) = 0;

	virtual GHOG_LIB_STATUS create_descriptor(cv::Mat gradients,
		cv::Size block_size,
		int num_bins,
		DescriptorCallback* callback) = 0;

	virtual GHOG_LIB_STATUS classify(cv::Mat img,
		ClassifyCallback* callback) = 0;

	virtual GHOG_LIB_STATUS locate(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride,
		LocateCallback* callback) = 0;

	virtual void resize_sync(cv::Mat image,
		cv::Size new_size,
		cv::Mat& resized_image) = 0;

	virtual void calc_gradient_sync(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase) = 0;

	virtual void create_descriptor_sync(cv::Mat gradients,
		cv::Size block_size,
		int num_bins,
		cv::Mat& descriptor) = 0;

	virtual void load_settings(std::string filename) = 0;

	virtual void set_classifier(IClassifier* classifier) = 0;

	virtual GHOG_LIB_STATUS set_img_resize(cv::Size img_resize) = 0;
	virtual cv::Size get_img_resize() = 0;
	virtual GHOG_LIB_STATUS set_num_bins(int num_bins) = 0;
	virtual int get_num_bins() = 0;
	virtual GHOG_LIB_STATUS set_block_size(cv::Size block_size) = 0;
	virtual cv::Size get_block_size() = 0;
};

} /* namespace lib */
} /* namespace ghog */

#endif /* IHOG_H_ */
