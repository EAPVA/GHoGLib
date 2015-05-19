/*
 * GHogDescriptor.cpp
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#include <include/GHogDescriptor.h>

namespace ghog
{
namespace lib
{

GHogDescriptor::~GHogDescriptor()
{
	// TODO Auto-generated destructor stub
}

Histogram GHogDescriptor::get_histogram(int num_hist)
{
	return _cell.at(num_hist);
}

cv::Mat GHogDescriptor::get_values()
{
	cv::Mat ret(_cell.size(), _cell[0].get_num_of_bins(), CV_32FC1);
	for (int i = 0; i < _cell.size(); ++i)
	{
		_cell[i].get_hist().copyTo(ret.row(i));
	}
	ret.reshape(1, 1);
	return ret;
}

} /* namespace lib */
} /* namespace ghog */
