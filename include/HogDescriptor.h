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

#include <include/HogCallbacks.h>
#include <include/IClassifier.h>
#include <include/Settings.h>

namespace ghog
{
namespace lib
{

class HogDescriptor
{
public:
	HogDescriptor(std::string settings_file);
	virtual ~HogDescriptor();

	virtual GHOG_LIB_STATUS alloc_buffer(cv::Size buffer_size,
		int type,
		cv::Mat& buffer,
		int padding_size);

	virtual GHOG_LIB_STATUS image_normalization(cv::Mat& image,
		ImageCallback* callback);
	virtual GHOG_LIB_STATUS image_normalization_sync(cv::Mat& image);

	virtual GHOG_LIB_STATUS calc_gradient(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase,
		GradientCallback* callback);
	virtual GHOG_LIB_STATUS calc_gradient_sync(cv::Mat input_img,
		cv::Mat& magnitude,
		cv::Mat& phase);

	virtual GHOG_LIB_STATUS create_descriptor(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor,
		DescriptorCallback* callback);
	virtual GHOG_LIB_STATUS create_descriptor_sync(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor);
	virtual GHOG_LIB_STATUS create_descriptor_sync(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat& descriptor,
		cv::Mat& histograms);

	virtual GHOG_LIB_STATUS classify(cv::Mat img,
		ClassifyCallback* callback);
	virtual bool classify_sync(cv::Mat img);

	virtual GHOG_LIB_STATUS locate(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride,
		LocateCallback* callback);
	virtual std::vector<cv::Rect> locate_sync(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride);

	void load_settings(std::string filename);

	void set_classifier(IClassifier* classifier);

	GHOG_LIB_STATUS set_param(std::string param,
		std::string value);
	std::string get_param(std::string param);

	int get_descriptor_size();

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

	void calc_histogram(cv::Mat magnitude,
		cv::Mat phase,
		cv::Mat cell_histogram);

	void normalize_blocks(cv::Mat& descriptor);

	std::string get_module(std::string param_name);

	Settings _settings;

	IClassifier* _classifier;

	int _num_bins;
	cv::Size _cell_size; //In number of pixels
	cv::Size _block_size; //In number of cells
	cv::Size _block_stride; //In number of cells
	cv::Size _cell_grid; //In number of cells
	cv::Size _window_size; //In number of pixels

	GHOG_LIB_NORM_TYPE _norm_type;
	cv::Mat _gaussian_window;
};

} /* namespace lib */
} /* namespace ghog */

#endif /* HOGCPU_H_ */
