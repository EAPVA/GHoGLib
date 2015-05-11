/*
 * Histogram.cpp
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#include <include/Histogram.h>

namespace ghog
{
namespace lib
{

Histogram::Histogram(int num_bins,
	cv::Mat gradients)
{
	_bin_size = 360.0f / (float)num_bins;
	_bin_list = cv::Mat(1, num_bins, CV_32FC1, 0.0f);
}

Histogram::~Histogram()
{
	// TODO Auto-generated destructor stub
}

void Histogram::slide_window(cv::Mat delta_plus,
	cv::Mat delta_minus)
{

}

cv::Mat Histogram::get_hist()
{
	return _bin_list;
}

float Histogram::get_bin(int bin_pos)
{
	return _bin_list.at< float >(bin_pos);
}

float Histogram::get_min_bin_val(int bin_pos)
{
	return _bin_size * bin_pos;
}

float Histogram::get_max_bin_val(int bin_pos)
{
	return _bin_size * (bin_pos + 1);
}

float Histogram::get_mid_bin_val(int bin_pos)
{
	return get_min_bin_val(bin_pos) + (_bin_size / 2.0f);
}

} /* namespace lib */
} /* namespace ghog */
