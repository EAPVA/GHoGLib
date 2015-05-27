/*
 * Hog.h
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#ifndef HOG_H_
#define HOG_H_

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <include/GHogLibConstants.inc>
#include <include/HogCallbacks.h>
#include <include/MultilayerPerceptron.h>
#include <include/Settings.h>

namespace ghog
{
namespace lib
{

class Hog
{
public:
	Hog(std::string settings_file);
	virtual ~Hog();

	GHOG_LIB_STATUS resize(cv::Mat image,
		cv::Size new_size,
		ImageCallback* callback);

	GHOG_LIB_STATUS calc_gradient(cv::Mat input_img,
		ImageCallback* callback);

	GHOG_LIB_STATUS create_descriptor(cv::Mat gradients,
		cv::Size block_size,
		int num_bins,
		DescriptorCallback* callback);

	GHOG_LIB_STATUS classify(cv::Mat img,
		ClassifyCallback* callback);

	GHOG_LIB_STATUS locate(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride,
		LocateCallback* callback);

	void load_settings(std::string filename);

	void set_classifier(IClassifier* classifier);

	GHOG_LIB_STATUS set_img_resize();
	cv::Size get_img_resize();
	GHOG_LIB_STATUS set_num_bins();
	int get_num_bins();
	GHOG_LIB_STATUS set_block_size();
	cv::Size get_block_size();

protected:
	void resize_async(cv::Mat image,
		cv::Size new_size,
		ImageCallback* callback);

	void calc_gradient_async(cv::Mat input_img,
		ImageCallback* callback);

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

	void resize_impl(cv::Mat image,
		cv::Size new_size,
		cv::Mat& resized);

	void calc_gradient_impl(cv::Mat input_img,
		cv::Mat& gradients);

	void create_descriptor_impl(cv::Mat gradients,
		cv::Size block_size,
		int num_bins,
		cv::Mat& descriptor);

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
#endif /* HOG_H_ */
