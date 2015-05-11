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

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

class MultilayerPerceptron
{
public:
	MultilayerPerceptron(cv::Mat layers,
		float learning_rate = 0.2f,
		bool random_weights = true);
	virtual ~MultilayerPerceptron();

	cv::Mat feed_forward(cv::Mat input);
	void backpropagation(cv::Mat expected,
		cv::Mat actual);

	void update_weights();

	void train(cv::Mat inputs,
		cv::Mat expected_outputs);

	void train_multiple_times(cv::Mat inputs,
		cv::Mat expected_outputs,
		int num_times);

private:

	float activation(float sum);
	float activation_derivative(float sum);

	cv::Mat _layers;
	std::vector< cv::Mat > _weights;
	cv::Mat _biases;

	cv::Mat _last_sums;
	cv::Mat _last_results;
	cv::Mat _error_signals;
	cv::Mat _last_gradients;

	float _learning_rate;

	boost::random::mt19937 _random_gen;

	int _max_layer_size;
};

#endif /* MULTILAYERPERCEPTRON_H_ */
