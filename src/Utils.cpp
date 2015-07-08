/*
 * Utils.cpp
 *
 *  Created on: Jun 18, 2015
 *      Author: teider
 */

#include <include/Utils.h>

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

	if(ret.width % numerator.width)
	{
		ret.width++;
	}

	if(ret.height % numerator.height)
	{
		ret.height++;
	}

	return ret;
}

} /* namespace lib */
} /* namespace ghog */

