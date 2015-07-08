/*
 * GHogLibConstants.h
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#ifndef GHOGLIBCONSTANTS_H_
#define GHOGLIBCONSTANTS_H_

#include <opencv2/core/core.hpp>

namespace ghog
{
namespace lib
{

enum GHOG_LIB_STATUS
{
	GHOG_LIB_STATUS_OK = 0,
	GHOG_LIB_STATUS_INVALID_PARAMETER_NAME = -1,
	GHOG_LIB_STATUS_UNKNOWN_ERROR = -255
};

const int GHOG_LIB_DEFAULT_NUM_BINS = 9;
const cv::Size GHOG_LIB_DEFAULT_CELL_SIZE = cv::Size(8, 8);
const cv::Size GHOG_LIB_DEFAULT_BLOCK_SIZE = cv::Size(2, 2);
const cv::Size GHOG_LIB_DEFAULT_BLOCK_STRIDE = cv::Size(1, 1);
const cv::Size GHOG_LIB_DEFAULT_WIN_SIZE = cv::Size(8, 16);

} /* namespace lib */
} /* namespace ghog */

#endif /* GHOGLIBCONSTANTS_H_ */
