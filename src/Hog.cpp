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

Hog::Hog(std::string settings_file) :
	_settings(settings_file)
{
	_classifier = NULL;

	_img_resize.width = _settings.load_int("Hog",
		"CLASSIFICATION_IMAGE_HEIGHT");
	_img_resize.width = _settings.load_int("Hog", "CLASSIFICATION_IMAGE_WIDTH");

	_num_bins = _settings.load_int(std::string("Descriptor"), "NUMBER_OF_BINS");
	_block_size.width = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_COLS");
	_block_size.height = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_ROWS");
}

Hog::~Hog()
{
// TODO Auto-generated destructor stub
}

GHOG_LIB_STATUS Hog::resize(cv::Mat image,
	cv::Size new_size,
	ImageCallback* callback)
{
	boost::thread(&Hog::resize_impl, this, image, new_size, callback)
		.detach();
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS Hog::calc_gradient(cv::Mat input_img,
	ImageCallback* callback)
{
	boost::thread(&Hog::calc_gradient_impl, this, input_img, callback)
		.detach();
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS Hog::classify(cv::Mat img,
	ClassifyCallback* callback)
{
	boost::thread(&Hog::classify_impl, this, img, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS Hog::locate(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride,
	LocateCallback* callback)
{
	boost::thread(&Hog::locate_impl, this, img, roi, window_size, window_stride,
		callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void Hog::set_classifier(IClassifier* classifier)
{
	_classifier = classifier;
}

void Hog::resize_impl(cv::Mat image,
	cv::Size new_size,
	ImageCallback* callback)
{
	cv::Mat ret;
	cv::resize(image, ret, new_size, 0, 0, CV_INTER_AREA);
	callback->image_processed(image, ret);
}

void Hog::calc_gradient_impl(cv::Mat input_img,
	ImageCallback* callback)
{
	cv::Mat ret;
	cv::Mat grad[2];
	cv::Sobel(input_img, grad[0], 1, 1, 0, 1);
	cv::Sobel(input_img, grad[1], 1, 0, 1, 1);
	cv::cartToPolar(grad[0], grad[1], grad[0], grad[1], true);
	cv::merge(grad, 2, ret);
	callback->image_processed(input_img, ret);
}

void Hog::classify_impl(cv::Mat img,
	ClassifyCallback* callback)
{
}

void Hog::locate_impl(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride,
	LocateCallback* callback)
{
	std::vector< cv::Rect > ret;
	callback->objects_located(img, ret);
}

} /* namespace lib */
} /* namespace ghog */
