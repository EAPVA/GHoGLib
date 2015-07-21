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
/**
 * \brief Callback to be used when a image processing operation finishes.
 */
class ImageCallback
{
public:
	virtual ~ImageCallback()
	{
	}
	/**
	 * \brief Called when the image is processed completely.
	 *
	 * \param processed The processed image.
	 */
	virtual void image_processed(cv::Mat processed) = 0;
};

/**
 * \brief Callback to be used when a gradient calculation operation finishes.
 */
class GradientCallback
{
public:
	virtual ~GradientCallback()
	{
	}
	/**
	 * \brief Called when the gradients are calculated completely.
	 *
	 * \param gradients_magnitude A 2D buffer with the gradients magnitudes.
	 * \param gradients_phase A 2D buffer with the gradients orientations.
	 */
	virtual void gradients_obtained(cv::Mat gradients_magnitude,
		cv::Mat gradients_phase) = 0;
};

/**
 * \brief Callback to be used when a descriptor calculation operation finishes.
 */
class DescriptorCallback
{
public:
	virtual ~DescriptorCallback()
	{
	}
	/**
	 * \brief Called when the descriptor is calculated completely.
	 *
	 * \param descriptor The resulting descriptor.
	 */
	virtual void descriptor_obtained(cv::Mat descriptor) = 0;
};

/**
 * \brief Callback to be used when a classification operation finishes.
 */
class ClassifyCallback
{
public:
	virtual ~ClassifyCallback()
	{
	}
	/**
	 * \brief Called when a classification finishes.
	 *
	 * \param positive True if the object matches positive, false otherwise.
	 */
	virtual void classification_result(bool positive) = 0;
};

/**
 * \brief Callback to be used when a detection operation finishes.
 */
class LocateCallback
{
public:
	virtual ~LocateCallback()
	{
	}
	/**
	 * \brief Called when a detection finishes.
	 *
	 * \param found_objects A list of bounding rectangles on found object on the image.
	 */
	virtual void objects_located(std::vector< cv::Rect > found_objects) = 0;
};

} /* namespace lib */
} /* namespace ghog */
#endif /* HOGCALLBACKS_H_ */
