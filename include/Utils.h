/*
 * Utils.h
 *
 *  Created on: Jun 18, 2015
 *      Author: teider
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <opencv2/core/core.hpp>

namespace ghog
{
namespace lib
{

class Utils
{
public:
	virtual ~Utils()
	{
	}

	static cv::Size partition(cv::Size numerator,
		cv::Size denominator);
};

} /* namespace lib */
} /* namespace ghog */

#endif /* UTILS_H_ */
