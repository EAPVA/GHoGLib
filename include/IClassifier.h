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

class ClassificationCallback
{
public:
	virtual ~ClassificationCallback() = 0;
	virtual void result(cv::Mat inputs,
		cv::Mat output) = 0;
};

class TrainingCallback
{
public:
	virtual ~TrainingCallback() = 0;
	virtual void finished(cv::Mat train_data) = 0;
};

class IClassifier
{
public:
	virtual ~IClassifier()
	{
	}

	virtual GHOG_LIB_STATUS train_async(cv::Mat train_data,
		cv::Mat expected_outputs,
		TrainingCallback* callback) = 0;
	virtual GHOG_LIB_STATUS train_sync(cv::Mat train_data,
		cv::Mat expected_outputs) = 0;

	virtual GHOG_LIB_STATUS classify_async(cv::Mat input,
		ClassificationCallback* callback) = 0;
	virtual cv::Mat classify_sync(cv::Mat input) = 0;

	virtual GHOG_LIB_STATUS load(std::string filename) = 0;
	virtual GHOG_LIB_STATUS save(std::string filename) = 0;

	virtual GHOG_LIB_STATUS set_parameter(std::string parameter,
		std::string value) = 0;
	virtual std::string get_parameter(std::string parameter) = 0;
};

} /* namespace lib */
} /* namespace ghog */
#endif /* ICLASSIFIER_H_ */
