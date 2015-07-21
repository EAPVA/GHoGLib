/*
 * IClassifier.h
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#ifndef ICLASSIFIER_H_
#define ICLASSIFIER_H_

#include <string>

#include <opencv2/core/core.hpp>

#include <include/GHogLibConstants.inc>

namespace ghog
{
namespace lib
{

/**
 * \brief Callback to be used when a classification operation finishes.
 */
class ClassificationCallback
{
public:
	virtual ~ClassificationCallback() = 0;
	/**
	 * \brief Called when a classification finishes.
	 *
	 * \todo Change the input image to an identifier (will be easier for the
	 * user to check).
	 *
	 * \param inputs The original input image.
	 * \param output A row vector containing the output.
	 */
	virtual void result(cv::Mat inputs,
		cv::Mat output) = 0;
};

/**
 * \brief Callback to be called when a training operation finishes.
 */
class TrainingCallback
{
public:
	virtual ~TrainingCallback() = 0;
	/**
	 * \brief Called when the training finishes.
	 *
	 * \todo Remove the training_data parameter, maybe replace with an
	 * identifier (not as necessary on this operation).
	 *
	 * \param train_data The training set passed.
	 */
	virtual void finished(cv::Mat train_data) = 0;
};

/**
 * \brief Interface for a classifier.
 *
 * Can be used by some internal functions, must be able to perform training,
 * classification, configuration and to save/load the current state (including
 * configuration and training).
 */
class IClassifier
{
public:
	virtual ~IClassifier()
	{
	}

	/**
	 * \brief Asynchronous supervised training.
	 *
	 * \param train_data A 2D buffer with one training instance per row.
	 * \param expected_outputs A row vector with the corresponding expected
	 * output for each of the training instance (column i of the vector is the
	 * expected output of row i of the training set).
	 * \param callback Callback object to be called when the function finishes.
	 */
	virtual GHOG_LIB_STATUS train_async(cv::Mat train_data,
		cv::Mat expected_outputs,
		TrainingCallback* callback) = 0;
	/**
	 * \brief Synchronous supervised training.
	 *
	 * \param train_data A 2D buffer with one training instance per row.
	 * \param expected_outputs A row vector with the corresponding expected
	 * output for each of the training instance (column i of the vector is the
	 * expected output of row i of the training set).
	 */
	virtual void train_sync(cv::Mat train_data,
		cv::Mat expected_outputs) = 0;
	/**
	 * \brief Asynchronous classification.
	 *
	 * \param input A row vector containing one descriptor.
	 * \param callback Callback object to be called when the function finishes.
	 */
	virtual GHOG_LIB_STATUS classify_async(cv::Mat input,
		ClassificationCallback* callback) = 0;
	/**
	 * \brief Synchronous classification.
	 *
	 * \param input A row vector containing one descriptor.
	 *
	 * \returns A row vector containing the output. Most classifiers will return
	 * a single value, but Neural Networks, for example, can return more than
	 * one value on the output.
	 */
	virtual cv::Mat classify_sync(cv::Mat input) = 0;

	/**
	 * \brief Loads classifier from file.
	 *
	 * \param filename Relative path to file.
	 */
	virtual GHOG_LIB_STATUS load(std::string filename) = 0;
	/**
	 * \brief Saves classifier into file.
	 *
	 * \param filename Relative path to file.
	 */
	virtual GHOG_LIB_STATUS save(std::string filename) = 0;

	/**
	 * \brief Sets a configuration parameter of the classifier.
	 *
	 * \param parameter Name of the parameter
	 * \param value New value for the parameter.
	 */
	virtual GHOG_LIB_STATUS set_parameter(std::string parameter,
		std::string value) = 0;
	/**
	 * \brief Reads a configuration parameter of the classifier.
	 *
	 * \param parameter Name of the parameter.
	 *
	 * \returns Current value of the parameter.
	 */
	virtual std::string get_parameter(std::string parameter) = 0;
};

} /* namespace lib */
} /* namespace ghog */
#endif /* ICLASSIFIER_H_ */
