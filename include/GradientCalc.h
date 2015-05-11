/*
 * GradientCalc.h
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#ifndef GRADIENTCALC_H_
#define GRADIENTCALC_H_

#include <include/GHogLibConstants.h>

#include <opencv2/core/core.hpp>

namespace ghog
{
namespace lib
{

class GradientCallback
{
public:
	virtual ~GradientCallback() = 0;
	virtual void operator()(cv::Mat magnitude,
		cv::Mat phase) = 0;
};

GHOG_LIB_STATUS calc_gradient(cv::Mat input_img,
	GradientCallback* callback);

} /* namespace lib */
} /* namespace ghog */

#endif /* GRADIENTCALC_H_ */
