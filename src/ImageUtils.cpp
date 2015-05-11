/*
 * ImageUtils.cpp
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#include <include/ImageUtils.h>

namespace ghog
{
namespace lib
{

ImageUtils::ImageUtils(ImageCallback* callback) :
	_callback(callback)
{
	// TODO Auto-generated constructor stub

}

ImageUtils::~ImageUtils()
{
	// TODO Auto-generated destructor stub
}

GHOG_LIB_STATUS ImageUtils::resize(cv::Mat image)
{
	return GHOG_LIB_STATUS_OK;
}

} /* namespace lib */
} /* namespace ghog */
