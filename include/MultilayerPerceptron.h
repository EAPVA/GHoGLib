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

#include <include/IClassifier.h>

namespace ghog
{

namespace lib
{

class MultilayerPerceptron: public IClassifier
{
public:
	MultilayerPerceptron(cv::Mat layers,
		float learning_rate = 0.2f,
		bool random_weights = true);
	virtual ~MultilayerPerceptron();

	void train_async(cv::Mat inputs,
		cv::Mat expected_outputs);
	virtual void classify_async(cv::Mat input);

	void load(std::string filename);
	void save(std::string filename);

	void set_parameter(std::string parameter,
		std::string value);
	std::string get_parameter(std::string parameter);

	cv::Mat feed_forward(cv::Mat input);
	void backpropagation(cv::Mat expected,
		cv::Mat actual);

	void update_weights();

	void train_multiple_times(cv::Mat inputs,
		cv::Mat expected_outputs,
		int num_times);

protected:

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

} /* namespace lib */
} /* namespace ghog */
#endif /* MULTILAYERPERCEPTRON_H_ */
