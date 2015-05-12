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

namespace ghog
{
namespace lib
{

class ClassifierCallback
{
	virtual ~ClassifierCallback() = 0;
	virtual void operator()(cv::Mat output) = 0;
};

class IClassifier
{
public:
	virtual ~IClassifier() = 0;

	virtual void train(cv::Mat train_data,
		cv::Mat expected_outputs) = 0;
	virtual void classify(cv::Mat input) = 0;

	virtual void load(std::string filename) = 0;
	virtual void save(std::string filename) = 0;

	virtual void set_parameter(std::string parameter,
		std::string value);
	virtual std::string get_parameter(std::string parameter);

protected:
	ClassifierCallback* _callback;
};

} /* namespace lib */
} /* namespace ghog */
#endif /* ICLASSIFIER_H_ */