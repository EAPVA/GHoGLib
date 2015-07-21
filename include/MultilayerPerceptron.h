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

/**
 * \brief Neural Network implementation on CPU.
 */
class MultilayerPerceptron: public IClassifier
{
public:
	/**
	 * \brief XML config file constructor.
	 *
	 * \warning Not yet implemented.
	 *
	 * Constructs a MLP neural network based on the configurations provided by the
	 * XML config file specified. If the XML file can't be found, creates one
	 * using default parameters.
	 *
	 * \param settings_file Relative or absolute path to XML file.
	 */
	MultilayerPerceptron(std::string filename);
	/**
	 * \brief Multilayer Perceptron constructor.
	 *
	 * Constructs a MLP neural network based on the configurations passed in the
	 * parameters.
	 *
	 * \param layers A row vector containing the number of neurons on each layer.
	 * \param learning_rate The network's learning rate for learning.
	 * \param max_iterations The maximum number of iterations for training.
	 * \param target_error The The target error for training.
	 * \param random_weights If true, all weights and biases will be randomized.
	 * Else they will start at zero (heavily not recommended).
	 */
	MultilayerPerceptron(cv::Mat layers,
		float learning_rate = 0.2f,
		float target_error = 1e-6f,
		int max_iterations = 1000,
		bool random_weights = true);
	/**
	 * \brief Default destructor.
	 */
	virtual ~MultilayerPerceptron();

	GHOG_LIB_STATUS train_async(cv::Mat train_data,
		cv::Mat expected_outputs,
		TrainingCallback* callback);
	GHOG_LIB_STATUS classify_async(cv::Mat input,
		ClassificationCallback* callback);

	void train_sync(cv::Mat train_data,
		cv::Mat expected_outputs);
	cv::Mat classify_sync(cv::Mat input);

	GHOG_LIB_STATUS load(std::string filename);
	GHOG_LIB_STATUS save(std::string filename);

	virtual GHOG_LIB_STATUS set_parameter(std::string parameter,
		std::string value) = 0;
	virtual std::string get_parameter(std::string parameter) = 0;

protected:
	/**
	 * \brief Helper function for asynchronous operation.
	 *
	 * \bug Not using the max_iterations and target_error parameters.
	 *
	 * \see train_async(cv::Mat train_data,cv::Mat expected_outputs,TrainingCallback* callback)
	 */
	void train_async_impl(cv::Mat train_data,
		cv::Mat expected_outputs,
		TrainingCallback* callback);
	/**
	 * \brief Helper function for asynchronous operation.
	 *
	 * \see classify_async(cv::Mat input,ClassificationCallback* callback)
	 */
	void classify_async_impl(cv::Mat input,
		ClassificationCallback* callback);

	/**
	 * \brief Calculates the output of the neural network.
	 *
	 * The intermediate results are stored, for later use on training.
	 *
	 * \param input A row vector with network input.
	 *
	 * \returns A row vector with the resulting output of the network.
	 */
	cv::Mat feed_forward(cv::Mat input);
	/**
	 * \brief Calculates the error propagation of the network.
	 *
	 * Starting from the output layer, each layer calculates an error function
	 * and propagates the error backwards through the network. The intermediate
	 * results previously calculated are used here, and each intermediate error
	 * signal is also stored for later use.
	 *
	 * \param expected The expected output of the network (target to which it is
	 * being trained for).
	 * \param actual The output returned by the network (the output of the
	 * function feed_forward(cv::Mat input) with the current weights).
	 */
	void backpropagation(cv::Mat expected,
		cv::Mat actual);
	/**
	 * \brief Updates the network based on the last errors calculated.
	 */
	void update_weights();

	/**
	 * \brief Calculates the activation function of a neuron.
	 *
	 * For now only supports tanh.
	 *
	 * \todo Add support for other activation functions.
	 */
	float activation(float sum);
	/**
	 * \brief Calculates the derivative of the activation function.
	 *
	 * For now only supports tanh' = sech^2.
	 *
	 * \todo Add support for other activation functions.
	 */
	float activation_derivative(float sum);

	cv::Mat _layers; /**<Row vector containing network layer configuration. Each
	element represents the number of elements in the respective layer.*/
	std::vector< cv::Mat > _weights;/**<Weights of the network. Each matrix on
	the vector contains the weights for a different layer. Each row on the
	matrix contains the weights for a single neuron.*/
	cv::Mat _biases;/**<Biases of the network. Each row contains the bias for a
	different layer and each element of the row contains the bias for a different
	neuron.*/

	cv::Mat _last_sums; /**<Internal storage of the last non-activated outputs
	of each neuron*/
	cv::Mat _last_results;/**<Internal storage of the last activated outputs
	of each neuron*/
	cv::Mat _error_signals;/**<Internal storage of the last calculated error
	signals*/
	cv::Mat _last_gradients;/**<Internal storage of the last calculated error
	gradients*/

	float _learning_rate; /**< A constant multiplying the update to be added to
	the weights*/
	float _target_error; /**< A small constant representing the desired maximum
	error rate.*/
	int _max_iterations; /**< The maximum number of iterations to be run when
	training.*/

	MLPSettings _settings; /**<Responsible for XML read/write.*/
};

} /* namespace lib */
} /* namespace ghog */
#endif /* MULTILAYERPERCEPTRON_H_ */
