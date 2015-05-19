/*
 * GHogDescriptor.h
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#ifndef GHOGDESCRIPTOR_H_
#define GHOGDESCRIPTOR_H_

#include <opencv2/core/core.hpp>

#include <include/Histogram.h>

namespace ghog
{
namespace lib
{

class GHogDescriptor
{
public:
	GHogDescriptor(std::vector< Histogram > histogram_list) :
		_cell(histogram_list)
	{
	}
	virtual ~GHogDescriptor();

	Histogram get_histogram(int num_hist);
	cv::Mat get_values();

protected:
	std::vector< Histogram > _cell;
};

} /* namespace lib */
} /* namespace ghog */
#endif /* GHOGDESCRIPTOR_H_ */
