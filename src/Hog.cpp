/*
 * Hog.cpp
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#include <include/Hog.h>

#include <boost/thread.hpp>

namespace ghog
{
namespace lib
{

Hog::Hog(HogCallback* callback,
	std::string settings_file) :
	_callback(callback),
	_settings(settings_file)
{
	_classifier = NULL;


	_num_bins = 0;
	_grid_size = cv::Size(0, 0);
	_block_size = cv::Size(0, 0);
	_block_stride = cv::Size(0, 0);
}

GHOG_LIB_STATUS Hog::locate(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride)
{
	boost::thread(&Hog::locate_impl, this, img, roi, window_size, window_stride)
		.detach();
	return GHOG_LIB_STATUS_OK;
}

void Hog::locate_impl(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride)
{
	std::vector< cv::Rect > ret;
	_callback->objects_detected(ret);
}

void Hog::set_classifier(IClassifier* classifier)
{
	_classifier = classifier;
}

void Hog::set_callback(HogCallback* callback)
{
	_callback = callback;
}

Hog::~Hog()
{
	// TODO Auto-generated destructor stub
}

} /* namespace lib */
} /* namespace ghog */
