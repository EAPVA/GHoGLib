/*
 * ImageCallback.h
 *
 *  Created on: May 11, 2015
 *      Author: marcelo
 */

#ifndef HOGCALLBACKS_H_
#define HOGCALLBACKS_H_

#include <opencv2/core/core.hpp>

namespace ghog
{
namespace lib
{

class ImageCallback
{
public:
	virtual ~ImageCallback()
	{
	}
	virtual void image_processed(cv::Mat processed) = 0;
};

class GradientCallback
{
public:
	virtual ~GradientCallback()
	{
	}
	virtual void gradients_obtained(cv::Mat gradients_magnitude,
		cv::Mat gradients_phase) = 0;
};

class DescriptorCallback
{
public:
	virtual ~DescriptorCallback()
	{
	}
	virtual void descriptor_obtained(cv::Mat descriptor) = 0;
};

class ClassifyCallback
{
public:
	virtual ~ClassifyCallback()
	{
	}
	virtual void classification_result(bool positive) = 0;
};

class LocateCallback
{
public:
	virtual ~LocateCallback()
	{
	}
	virtual void objects_located(std::vector< cv::Rect > found_objects) = 0;
};

} /* namespace lib */
} /* namespace ghog */
#endif /* HOGCALLBACKS_H_ */
