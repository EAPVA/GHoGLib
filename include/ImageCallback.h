/*
 * ImageCallback.h
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#ifndef IMAGECALLBACK_H_
#define IMAGECALLBACK_H_

#include <opencv2/core/core.hpp>

namespace ghog
{
namespace lib
{

class ImageCallback
{
public:
	virtual ~ImageCallback() = 0;
	virtual void operator()(cv::Mat ret_mat) = 0;
};

} /* namespace lib */
} /* namespace ghog */
#endif /* IMAGECALLBACK_H_ */