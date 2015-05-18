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

#include <include/MultilayerPerceptron.h>
#include <include/GHogLibConstants.inc>
#include <include/Settings.h>

namespace ghog
{
namespace lib
{

class HogCallback
{
public:
	virtual ~HogCallback()
	{
	}
	virtual void objects_detected(cv::Mat img,
		std::vector< cv::Rect > found_objects) = 0;
	virtual void classification_result(cv::Mat img,
		bool positive) = 0;
};

class Hog
{
public:
	Hog(HogCallback* callback,
		std::string settings_file);
	virtual ~Hog();

	GHOG_LIB_STATUS classify(cv::Mat img);

	GHOG_LIB_STATUS locate(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride);

	void set_classifier(IClassifier* classifier);
	void set_callback(HogCallback* callback);

protected:
	void classify_impl(cv::Mat img);
	void locate_impl(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride);

	IClassifier* _classifier;
	HogCallback* _callback;
	Settings _settings;

	int _num_bins;
	cv::Size _grid_size; // In number of blocks
	cv::Size _block_size; // In number of cells
	cv::Size _block_stride; // In number of cells
};

} /* namespace lib */
} /* namespace ghog */
#endif /* HOG_H_ */
