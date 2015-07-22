/*
 * Utils.h
 *
 *  Created on: Jun 18, 2015
 *      Author: teider
 */

#ifndef GHOGLIB_UTILS_H_
#define GHOGLIB_UTILS_H_

#include <opencv2/core/core.hpp>

namespace ghog
{
namespace lib
{

/**
 * \brief General utilities.
 */
class Utils
{
public:
	virtual ~Utils()
	{
	}

	/**
	 * \brief Calculates ceil(numerator / denominator) for each dimension of the inputs.
	 *
	 * Currently not in use on the library.
	 */
	static cv::Size partition(cv::Size numerator,
		cv::Size denominator);
};

} /* namespace lib */
} /* namespace ghog */

#endif /* GHOGLIB_UTILS_H_ */
