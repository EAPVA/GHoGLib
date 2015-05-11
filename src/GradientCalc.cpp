/*
 * GradientCalc.cpp
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#include <include/GradientCalc.h>

namespace ghog
{
namespace lib
{

GradientCalc::GradientCalc(GradientCallback* callback) :
	_callback(callback)
{

}

GradientCalc::~GradientCalc()
{

}

GHOG_LIB_STATUS GradientCalc::calc_gradient(cv::Mat input_img)
{
	return GHOG_LIB_STATUS_OK;
}

void GradientCalc::calc_gradient_impl(cv::Mat input_img)
{
	(*_callback)(input_img, input_img);
}

}
/* namespace lib */
} /* namespace ghog */
