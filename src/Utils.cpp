/*
 * Utils.cpp
 *
 *  Created on: Jun 18, 2015
 *      Author: teider
 */

#include <include/Utils.h>

#include <iostream>

namespace ghog
{
namespace lib
{

cv::Size Utils::partition(cv::Size numerator,
	cv::Size denominator)
{
	cv::Size ret;
	ret.width = numerator.width / denominator.width;
	ret.height = numerator.height / denominator.height;

	if(numerator.width % denominator.width > 0)
	{
		ret.width++;
	}

	if(numerator.height % denominator.height > 0)
	{
		ret.height++;
	}

	return ret;
}

} /* namespace lib */
} /* namespace ghog */

