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
#include <include/IHog.h>

namespace ghog
{
namespace lib
{

class HogGPU: public IHog
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

	GHOG_LIB_STATUS create_descriptor(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor,
		DescriptorCallback* callback);

	void create_descriptor_sync(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor);

	GHOG_LIB_STATUS classify(cv::Mat img,
		ClassifyCallback* callback);

	bool classify_sync(cv::Mat img);

	GHOG_LIB_STATUS locate(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride,
		LocateCallback* callback);

	std::vector< cv::Rect > locate_sync(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride);

	void load_settings(std::string filename);

	void set_classifier(IClassifier* classifier);

	GHOG_LIB_STATUS set_param(std::string param,
		std::string value);
	std::string get_param(std::string param);

protected:
	void resize_async(cv::Mat image,
		cv::Size new_size,
		cv::Mat& resized_image,
		ImageCallback* callback);

	void calc_gradient_async(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase,
		GradientCallback* callback);

	void create_descriptor_async(cv::Mat gradients,
		cv::Size block_size,
		int num_bins,
		DescriptorCallback* callback);

	void classify_async(cv::Mat img,
		ClassifyCallback* callback);

	void locate_async(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride,
		LocateCallback* callback);

	void calc_histogram(cv::Mat gradients,
		int num_bins,
		cv::Mat histogram);

	Settings _settings;

	IClassifier* _classifier;

	cv::Size _img_resize;
	int _num_bins;
	cv::Size _block_size; // In number of cells
};

} /* namespace lib */
} /* namespace ghog */

#endif /* HOGGPU_H_ */
