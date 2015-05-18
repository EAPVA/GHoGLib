/*
 * ImageUtils.h
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#ifndef IMAGEUTILS_H_
#define IMAGEUTILS_H_

#include <include/GHogLibConstants.inc>
#include <include/ImageCallback.h>

namespace ghog
{
namespace lib
{

class ImageUtils
{
public:
	ImageUtils();
	virtual ~ImageUtils();

	GHOG_LIB_STATUS resize(cv::Mat image,
		cv::Size new_size,
		ImageCallback* callback);

protected:
	void resize_impl(cv::Mat image,
		cv::Size new_size,
		ImageCallback* callback);
};

} /* namespace lib */
} /* namespace ghog */
#endif /* IMAGEUTILS_H_ */
