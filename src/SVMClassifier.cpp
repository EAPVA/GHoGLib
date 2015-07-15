/*
 * SVMClassifier.cpp
 *
 *  Created on: Jul 8, 2015
 *      Author: teider
 */

#include <include/SVMClassifier.h>

#include <boost/thread.hpp>

namespace ghog
{
namespace lib
{

SVMClassifier::SVMClassifier()
{

}

SVMClassifier::~SVMClassifier()
{
	// TODO Auto-generated destructor stub
}

GHOG_LIB_STATUS SVMClassifier::train_async(cv::Mat train_data,
	cv::Mat expected_outputs,
	TrainingCallback* callback)
{
	boost::thread(&SVMClassifier::train_async_impl, this, train_data,
		expected_outputs, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void SVMClassifier::train_async_impl(cv::Mat train_data,
	cv::Mat expected_outputs,
	TrainingCallback* callback)
{
	train_sync(train_data, expected_outputs);
	callback->finished(train_data);
}

void SVMClassifier::train_sync(cv::Mat train_data,
	cv::Mat expected_outputs)
{
	_svm.train_auto(train_data, expected_outputs, cv::Mat(), cv::Mat(),
		_svm_params, 5);
}

GHOG_LIB_STATUS SVMClassifier::classify_async(cv::Mat input,
	ClassificationCallback* callback)
{
	boost::thread(&SVMClassifier::classify_async_impl, this, input, callback)
		.detach();
	return GHOG_LIB_STATUS_OK;
}

void SVMClassifier::classify_async_impl(cv::Mat input,
	ClassificationCallback* callback)
{
	cv::Mat ret = classify_sync(input);
	callback->result(input, ret);
}

cv::Mat SVMClassifier::classify_sync(cv::Mat input)
{
	cv::Mat ret(1, 1, CV_32FC1);
	ret.at< float >(0) = _svm.predict(input);
	return ret;
}

GHOG_LIB_STATUS SVMClassifier::load(std::string filename)
{
	_svm.load(filename.c_str());
	return GHOG_LIB_STATUS_OK;
}
GHOG_LIB_STATUS SVMClassifier::save(std::string filename)
{
	_svm.save(filename.c_str());
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS SVMClassifier::set_parameter(std::string parameter,
	std::string value)
{
	return GHOG_LIB_STATUS_OK;
}

std::string SVMClassifier::get_parameter(std::string parameter)
{
	return "";
}

} /* namespace lib */
} /* namespace ghog */
