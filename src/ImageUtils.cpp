/*
 * ImageUtils.cpp
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#include <include/ImageUtils.h>

#include <iostream>

#include <boost/thread.hpp>

#include <opencv2/imgproc/imgproc.hpp>

namespace ghog
{
namespace lib
{

ImageUtils::ImageUtils(ImageCallback* callback) :
	_callback(callback)
{
	// TODO Auto-generated constructor stub

}

ImageUtils::~ImageUtils()
{
	// TODO Auto-generated destructor stub
}

GHOG_LIB_STATUS ImageUtils::resize(cv::Mat image,
	cv::Size new_size)
{
	boost::thread(&ImageUtils::resize_impl, this, image, new_size).detach();
	return GHOG_LIB_STATUS_OK;
}

void ImageUtils::resize_impl(cv::Mat image,
	cv::Size new_size)
{
	cv::Mat ret;
	cv::resize(image, ret, new_size, 0, 0, CV_INTER_AREA);
	_callback->image_processed(ret);
}

} /* namespace lib */
} /* namespace ghog */
