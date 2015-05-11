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

} /* namespace lib */
} /* namespace ghog */
