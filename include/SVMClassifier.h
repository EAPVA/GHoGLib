/*
 * SVMClassifier.h
 *
 *  Created on: Jul 8, 2015
 *      Author: teider
 */

#ifndef SVMCLASSIFIER_H_
#define SVMCLASSIFIER_H_

#include <include/IClassifier.h>

#include <include/GHogLibConstants.inc>

#include <opencv2/ml/ml.hpp>

namespace ghog
{
namespace lib
{

class SVMClassifier: public IClassifier
{
public:
	SVMClassifier();
	virtual ~SVMClassifier();

	GHOG_LIB_STATUS train_async(cv::Mat train_data,
		cv::Mat expected_outputs,
		TrainingCallback* callback);
	void train_sync(cv::Mat train_data,
		cv::Mat expected_outputs);

	GHOG_LIB_STATUS classify_async(cv::Mat input,
		ClassificationCallback* callback);
	cv::Mat classify_sync(cv::Mat input);

	GHOG_LIB_STATUS load(std::string filename);
	GHOG_LIB_STATUS save(std::string filename);

	GHOG_LIB_STATUS set_parameter(std::string parameter,
		std::string value);
	std::string get_parameter(std::string parameter);

private:
	void train_async_impl(cv::Mat train_data,
		cv::Mat expected_outputs,
		TrainingCallback* callback);
	void classify_async_impl(cv::Mat input,
		ClassificationCallback* callback);

	CvSVM _svm;
	CvSVMParams _svm_params;
};

} /* namespace lib */
} /* namespace ghog */
#endif /* SVMCLASSIFIER_H_ */
