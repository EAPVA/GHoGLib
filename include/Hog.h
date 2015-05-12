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
#include <include/GHogLibConstants.h>
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
	virtual void operator()(std::vector< cv::Rect > found_objects) = 0;
};

class Hog
{
public:
	Hog(HogCallback* callback,
		std::string settings_file);
	virtual ~Hog();

	GHOG_LIB_STATUS locate(cv::Mat img,
		cv::Rect roi,
		cv::Size window_size,
		cv::Size window_stride);

	void set_classifier(IClassifier* classifier);

protected:
	IClassifier* _classifier;
	HogCallback* _callback;
	Settings _settings;

	cv::Rect _roi;
	cv::Size _window_size; // In pixels
	cv::Size _window_stride; // In pixels

	int _num_bins;
	cv::Size _grid_size; // In number of cells
	cv::Size _block_size; // In number of cells
	cv::Size _block_stride; // In number of cells
};

} /* namespace lib */
} /* namespace ghog */
#endif /* HOG_H_ */
