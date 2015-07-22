/*
 * SVMClassifier.h
 *
 *  Created on: Jul 8, 2015
 *      Author: teider
 */

#ifndef GHOGLIB_SVMCLASSIFIER_H_
#define GHOGLIB_SVMCLASSIFIER_H_

#include <include/IClassifier.h>

#include <include/GHogLibConstants.inc>

#include <opencv2/ml/ml.hpp>

namespace ghog
{
namespace lib
{
/**
 * \brief Sample SVM classifier. Currently uses OpenCV's SVM implementation.
 *
 * \todo Implement the configuration functions.
 */
class SVMClassifier: public IClassifier
{
public:
	/**
	 * \brief Default constructor.
	 */
	SVMClassifier();

	/**
	 * \brief Default destructor.
	 */
	virtual ~SVMClassifier();

	GHOG_LIB_STATUS train_async(cv::Mat train_data,
		cv::Mat expected_outputs,
		TrainingCallback* callback);
	void train_sync(cv::Mat train_data,
		cv::Mat expected_outputs);

	GHOG_LIB_STATUS classify_async(cv::Mat input,
		ClassificationCallback* callback);
	/**
	 * Synchronous classification.
	 *
	 * \param input A row vector containing one descriptor.
	 *
	 * \returns A row vector with a single element containing the output.
	 */
	cv::Mat classify_sync(cv::Mat input);

	GHOG_LIB_STATUS load(std::string filename);
	GHOG_LIB_STATUS save(std::string filename);

	/**
	 * Sets a configuration parameter of the classifier.
	 *
	 * \warning Not yet implemented.
	 *
	 * \param parameter Name of the parameter
	 * \param value New value for the parameter.
	 */
	GHOG_LIB_STATUS set_parameter(std::string parameter,
		std::string value);
	/**
	 * Reads a configuration parameter of the classifier.
	 *
	 * \warning Not yet implemented.
	 *
	 * \param parameter Name of the parameter.
	 *
	 * \returns Current value of the parameter.
	 */
	std::string get_parameter(std::string parameter);

private:
	/**
	 * \brief Helper function for asynchronous operation.
	 *
	 * \see train_async(cv::Mat train_data,cv::Mat expected_outputs,TrainingCallback* callback)
	 */
	void train_async_impl(cv::Mat train_data,
		cv::Mat expected_outputs,
		TrainingCallback* callback);
	/**
	 * \brief Helper function for asynchronous operation.
	 *
	 * \see GHOG_LIB_STATUS classify_async(cv::Mat input,ClassificationCallback* callback)
	 */
	void classify_async_impl(cv::Mat input,
		ClassificationCallback* callback);

	CvSVM _svm; /**<Internal SVM Classifier. */
	CvSVMParams _svm_params; /**<SVM parameters */
};

} /* namespace lib */
} /* namespace ghog */
#endif /* GHOGLIB_SVMCLASSIFIER_H_ */
