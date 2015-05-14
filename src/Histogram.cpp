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

	//TODO Move this to a Builder class later on
	//TODO Do this more efficiently
	cv::Mat aux[2];
	cv::split(gradients, aux);
	cv::Mat mag = aux[0];
	cv::Mat phase = aux[1];

	int left_bin, right_bin;
	float delta;

	float mag_total = 0.0f;

	for(int i = 0; i < mag.rows; ++i)
	{
		for(int j = 0; j < mag.cols; ++j)
		{
			if(mag.at< float >(i, j) > 0)
			{
				left_bin = find_bin(phase.at< float >(i, j));
				if(left_bin < 0)
					left_bin += get_num_of_bins();
				right_bin = (left_bin + 1) % get_num_of_bins();

				delta = (phase.at< float >(i, j) / _bin_size) - right_bin;
				if(delta > 1.0)
					delta -= get_num_of_bins();

				_bin_list.at< float >(left_bin) += (0.5 - delta)
					* mag.at< float >(i, j);
				_bin_list.at< float >(right_bin) += (0.5 + delta)
					* mag.at< float >(i, j);
			}
		}
	}
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

float Histogram::get_bin_size()
{
	return _bin_size;
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

float Histogram::get_num_of_bins()
{
	return _bin_list.cols;
}

int Histogram::find_bin(float phase_val)
{
	return (int)floor((phase_val - _bin_size / 2.0f) / _bin_size);
}

} /* namespace lib */
} /* namespace ghog */
