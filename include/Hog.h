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

	GHOG_LIB_STATUS classify(cv::Mat img,
		ClassifyCallback* callback);

	GHOG_LIB_STATUS locate(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride,
		LocateCallback* callback);

	void set_classifier(IClassifier* classifier);

protected:
	void resize_impl(cv::Mat image,
			cv::Size new_size,
			ImageCallback* callback);

	void calc_gradient_impl(cv::Mat input_img,
			ImageCallback* callback);

	void classify_impl(cv::Mat img,
		ClassifyCallback* callback);

	void locate_impl(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride,
		LocateCallback* callback);

	IClassifier* _classifier;
	Settings _settings;

	cv::Size _img_resize;
	int _num_bins;
	cv::Size _block_size; // In number of cells
};

} /* namespace lib */
} /* namespace ghog */
#endif /* HOG_H_ */
