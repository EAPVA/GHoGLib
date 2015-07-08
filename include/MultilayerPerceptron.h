/*
 * MultilayerPerceptron.h
 *
 *  Created on: Apr 28, 2015
 *      Author: teider
 */

#ifndef MULTILAYERPERCEPTRON_H_
#define MULTILAYERPERCEPTRON_H_

#include <vector>

#include <opencv2/core/core.hpp>

#include <include/IClassifier.h>
#include <include/MLPSettings.h>

namespace ghog
{

namespace lib
{

class MultilayerPerceptron: public IClassifier
{
public:
	MultilayerPerceptron(std::string filename);
	MultilayerPerceptron(cv::Mat layers,
		float learning_rate = 0.2f,
		float target_error = 1e-6f,
		int max_iterations = 1000,
		bool random_weights = true);
	virtual ~MultilayerPerceptron();

	GHOG_LIB_STATUS train_async(cv::Mat train_data,
		cv::Mat expected_outputs,
		TrainingCallback* callback) = 0;
	GHOG_LIB_STATUS classify_async(cv::Mat input,
		ClassificationCallback* callback) = 0;

	GHOG_LIB_STATUS train_sync(cv::Mat train_data,
		cv::Mat expected_outputs) = 0;
	cv::Mat classify_sync(cv::Mat input) = 0;

	GHOG_LIB_STATUS load(std::string filename);
	GHOG_LIB_STATUS save(std::string filename);

	virtual GHOG_LIB_STATUS set_parameter(std::string parameter,
		std::string value) = 0;
	virtual std::string get_parameter(std::string parameter) = 0;

protected:
	void train_async_impl(cv::Mat train_data,
		cv::Mat expected_outputs,
		TrainingCallback* callback);
	void classify_async_impl(cv::Mat input,
		ClassificationCallback* callback);

	cv::Mat feed_forward(cv::Mat input);
	void backpropagation(cv::Mat expected,
		cv::Mat actual);
	void update_weights();

	float activation(float sum);
	float activation_derivative(float sum);

	cv::Mat _layers;
	std::vector< cv::Mat > _weights;
	cv::Mat _biases;

	cv::Mat _last_sums;
	cv::Mat _last_results;
	cv::Mat _error_signals;
	cv::Mat _last_gradients;

	//For training
	float _learning_rate;
	float _target_error;
	int _max_iterations;

	MLPSettings _settings;
};

} /* namespace lib */
} /* namespace ghog */
#endif /* MULTILAYERPERCEPTRON_H_ */
