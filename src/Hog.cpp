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

	_num_bins = _settings.load_int(std::string("Descriptor"), "NUMBER_OF_BINS");
	_grid_size.width = _settings.load_int(std::string("Descriptor"),
		"GRID_SIZE_COLS");
	_grid_size.height = _settings.load_int(std::string("Descriptor"),
		"GRID_SIZE_ROWS");
	_block_size.width = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_COLS");
	_block_size.height = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_ROWS");
	_block_stride.width = _settings.load_int(std::string("Descriptor"),
		"BLOCK_STRIDE_COLS");
	_block_stride.height = _settings.load_int(std::string("Descriptor"),
		"BLOCK_STRIDE_ROWS");
}

GHOG_LIB_STATUS Hog::classify(cv::Mat img)
{
	boost::thread(&Hog::classify_impl, this, img).detach();
	return GHOG_LIB_STATUS_OK;
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

void Hog::classify_impl(cv::Mat img)
{

}

void Hog::locate_impl(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride)
{
	std::vector< cv::Rect > ret;
	_callback->objects_detected(img, ret);
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
