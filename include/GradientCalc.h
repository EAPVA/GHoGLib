/*
 * GradientCalc.h
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#ifndef GRADIENTCALC_H_
#define GRADIENTCALC_H_

#include <include/GHogLibConstants.inc>
#include <include/ImageCallback.h>

#include <opencv2/core/core.hpp>

namespace ghog
{
namespace lib
{

class GradientCalc
{
public:
	GradientCalc();
	~GradientCalc();

	GHOG_LIB_STATUS calc_gradient(cv::Mat input_img,
		ImageCallback* callback);

protected:

	void calc_gradient_impl(cv::Mat input_img,
		ImageCallback* callback);
};

} /* namespace lib */
} /* namespace ghog */

#endif /* GRADIENTCALC_H_ */
