/*
 * GradientCalc.cpp
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#include <include/GradientCalc.h>

#include <boost/thread.hpp>

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

void GradientCalc::calc_gradient_impl(cv::Mat input_img)
{
	_callback->image_processed(input_img);
}

} /* namespace lib */
} /* namespace ghog */
