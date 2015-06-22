/*
 * HogCPU.h
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#ifndef HOGCPU_H_
#define HOGCPU_H_

#include <vector>
#include <string>

#include <include/IClassifier.h>
#include <include/Settings.h>
#include <include/IHog.h>

namespace ghog
{
namespace lib
{

class HogDescriptor: public IHog
{
public:
	HogDescriptor(std::string settings_file);
	~HogDescriptor();

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
	void image_normalization_async(cv::Mat& image,
		ImageCallback* callback);

	void calc_gradient_async(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase,
		GradientCallback* callback);

	void create_descriptor_async(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor,
		DescriptorCallback* callback);

	void classify_async(cv::Mat img,
		ClassifyCallback* callback);

	void locate_async(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride,
		LocateCallback* callback);

	virtual void calc_histogram(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat cell_histogram);

	virtual void normalize_blocks(cv::Mat& descriptor);

	std::string get_module(std::string param_name);

	Settings _settings;

	IClassifier* _classifier;

	int _num_bins;
	cv::Size _block_size; //In number of cells
	cv::Size _block_stride; //In number of cells
	cv::Size _cell_size; //In number of pixels
};

} /* namespace lib */
} /* namespace ghog */

#endif /* HOGCPU_H_ */
