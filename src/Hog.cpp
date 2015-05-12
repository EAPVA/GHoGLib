/*
 * Hog.cpp
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#include <include/Hog.h>

namespace ghog
{
namespace lib
{

Hog::Hog(HogCallback* callback,
	std::string settings_file) :
	_callback(callback),
	_settings(settings_file)
{
	_classifier = NULL;
	_num_bins = 0;
	_grid_size = cv::Size(0,0);
	_block_size = cv::Size(0,0);
	_block_stride = cv::Size(0,0);
}

Hog::~Hog()
{
	// TODO Auto-generated destructor stub
}

} /* namespace lib */
} /* namespace ghog */
