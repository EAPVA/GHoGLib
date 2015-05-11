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
	virtual void operator()(cv::Mat gradients) = 0;
};

class GradientCalc
{
public:
	GradientCalc(GradientCallback* callback);
	~GradientCalc();

	GHOG_LIB_STATUS calc_gradient(cv::Mat input_img);

protected:
	GradientCallback* _callback;

	void calc_gradient_impl(cv::Mat input_img);
};

} /* namespace lib */
} /* namespace ghog */

#endif /* GRADIENTCALC_H_ */
