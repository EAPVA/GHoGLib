/*
 * GradientCalc.cpp
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#include <include/GradientCalc.h>

#include <boost/thread.hpp>

#include <opencv2/imgproc/imgproc.hpp>

namespace ghog
{
namespace lib
{

GradientCalc::GradientCalc(ImageCallback* callback) :
	_callback(callback)
{

}

GradientCalc::~GradientCalc()
{

}

GHOG_LIB_STATUS GradientCalc::calc_gradient(cv::Mat input_img)
{
	boost::thread(&GradientCalc::calc_gradient_impl, this, input_img).detach();
	return GHOG_LIB_STATUS_OK;
}

void GradientCalc::set_callback(ImageCallback* callback)
{
	_callback = callback;
}

void GradientCalc::calc_gradient_impl(cv::Mat input_img)
{
	cv::Mat ret;
	cv::Mat grad[2];
	cv::Sobel(input_img, grad[0], 1, 1, 0, 1);
	cv::Sobel(input_img, grad[1], 1, 0, 1, 1);
	cv::cartToPolar(grad[0], grad[1], grad[0], grad[1], true);
	cv::merge(grad, 2, ret);
	_callback->image_processed(ret);
}

} /* namespace lib */
} /* namespace ghog */
