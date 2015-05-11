/*
 * Histogram.h
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include <opencv2/core/core.hpp>

namespace ghog
{
namespace lib
{

class Histogram
{
public:
	Histogram(int num_bins,
		cv::Mat gradients);
	virtual ~Histogram();

	/**
	 * Slides histogram window, by adding some elements and subtracting others
	 */
	void slide_window(cv::Mat delta_plus,
		cv::Mat delta_minus);

	cv::Mat get_hist();
	float get_bin(int bin_pos);

	float get_bin_size();
	float get_min_bin_val(int bin_pos);
	float get_max_bin_val(int bin_pos);
	float get_mid_bin_val(int bin_pos);

protected:
	cv::Mat _bin_list;
	float _bin_size;
};

} /* namespace lib */
} /* namespace ghog */
#endif /* HISTOGRAM_H_ */
