/*
 * Neural.cpp
 *
 *  Created on: Apr 28, 2015
 *      Author: teider
 */

#include <include/MultilayerPerceptron.h>

ghog::lib::MultilayerPerceptron::MultilayerPerceptron(cv::Mat layers,
	float learning_rate,
	bool random_weights) :
	_layers(layers),
	_learning_rate(learning_rate)
{
	int num_layers = _layers.cols;
	_max_layer_size = 0;
	int* layer_ptr = _layers.ptr< int >(0);
	for(int l = 0; l < num_layers; ++l)
	{
		if(layer_ptr[l] > _max_layer_size)
		{
			_max_layer_size = layer_ptr[l];
		}
	}
	num_layers--; //The input layer is not included on most matrices.
	for(int l = 0; l < num_layers; ++l)
	{
		_weights.push_back(
			cv::Mat(_max_layer_size, _max_layer_size, CV_32FC1, 0.0f));
	}
	_biases = cv::Mat(num_layers, _max_layer_size, CV_32FC1, 0.0f);

	_last_sums = cv::Mat(num_layers, _max_layer_size, CV_32FC1, 0.0f);
	_last_results = cv::Mat(num_layers + 1, _max_layer_size, CV_32FC1, 0.0f);
	_error_signals = cv::Mat(num_layers + 1, _max_layer_size, CV_32FC1, 0.0f);
	_last_gradients = cv::Mat(num_layers, _max_layer_size, CV_32FC1, 0.0f);

	if(random_weights)
	{
		boost::random::uniform_real_distribution< > dist(-1.0f, 1.0f);
		layer_ptr = _layers.ptr< int >(0);
		layer_ptr++; //Skip the input layer;
		for(int l = 0; l < num_layers; ++l)
		{
			float* biases_ptr = _biases.ptr< float >(l);
			for(int j = 0; j < layer_ptr[l]; ++j)
			{
				biases_ptr[j] = dist(_random_gen);
				float* weights_ptr = _weights[l].ptr< float >(j);
				//Use number of nodes of previous layer as number of inputs.
				for(int k = 0; k < layer_ptr[l - 1]; ++k)
				{
					weights_ptr[k] = dist(_random_gen);
				}
			}
		}
	}
}

ghog::lib::MultilayerPerceptron::~MultilayerPerceptron()
{
	// TODO Auto-generated destructor stub
}

void ghog::lib::MultilayerPerceptron::train(cv::Mat inputs,
	cv::Mat expected_outputs)
{
	cv::Mat output;
	for(int i = 0; i < inputs.rows; ++i)
	{
		output = feed_forward(inputs.row(i));
		backpropagation(expected_outputs.row(i), output);
		update_weights();
	}
}

void ghog::lib::MultilayerPerceptron::classify(cv::Mat input)
{

}

void ghog::lib::MultilayerPerceptron::load(std::string filename)
{

}

void ghog::lib::MultilayerPerceptron::save(std::string filename)
{

}

void ghog::lib::MultilayerPerceptron::set_parameter(std::string parameter,
	std::string value)
{

}

std::string ghog::lib::MultilayerPerceptron::get_parameter(std::string parameter)
{

}

cv::Mat ghog::lib::MultilayerPerceptron::feed_forward(cv::Mat input)
{
	float* input_ptr = input.ptr< float >(0);
	float* output_ptr = _last_results.ptr< float >(0);
	for(int i = 0; i < input.cols; ++i)
	{
		output_ptr[i] = input_ptr[i];
	}
	int* layer_ptr = _layers.ptr< int >(0);
	layer_ptr++; //Skip input layer;
	for(int l = 0; l < _weights.size(); ++l)
	{
		input_ptr = _last_results.ptr< float >(l);
		output_ptr = _last_results.ptr< float >(l + 1);
		float* sum_ptr = _last_sums.ptr< float >(l);
		float* bias_ptr = _biases.ptr< float >(l);
		for(int n = 0; n < layer_ptr[l]; ++n)
		{
			float* weight_ptr = _weights[l].ptr< float >(n);
			sum_ptr[n] = bias_ptr[n];
			//Use number of nodes of previous layer as number of inputs.
			for(int w = 0; w < layer_ptr[l - 1]; ++w)
			{
				sum_ptr[n] += weight_ptr[w] * input_ptr[w];
			}
			output_ptr[n] = activation(sum_ptr[n]);
		}
	}
	int num_outputs = _layers.at< int >(_layers.cols - 1);
	return _last_results.row(_last_results.rows - 1).colRange(0, num_outputs);
}

void ghog::lib::MultilayerPerceptron::backpropagation(cv::Mat expected,
	cv::Mat actual)
{
	cv::Mat error = actual - expected;
	float* error_input_ptr = error.ptr< float >(0);
	float* error_output_ptr = _error_signals.ptr< float >(
		_error_signals.rows - 1);
	for(int i = 0; i < error.cols; ++i)
	{
		error_output_ptr[i] = error_input_ptr[i];
	}
	int* layer_ptr = _layers.ptr< int >(0);
	layer_ptr++; //Skip input layer;
	for(int l = _weights.size() - 1; l >= 0; --l)
	{
		float* sum_ptr = _last_sums.ptr< float >(l);
		float* gradient_ptr = _last_gradients.ptr< float >(l);
		error_input_ptr = _error_signals.ptr< float >(l + 1);
		error_output_ptr = _error_signals.ptr< float >(l);
		//Clear the error signal from the previous layer
		for(int n = 0; n < layer_ptr[l - 1]; ++n)
		{
			error_output_ptr[n] = 0.0f;
		}
		for(int n = 0; n < layer_ptr[l]; ++n)
		{
			float* weight_ptr = _weights[l].ptr< float >(n);
			gradient_ptr[n] = error_input_ptr[n]
				* activation_derivative(sum_ptr[n]);
			//Use number of nodes of previous layer as number of inputs.
			for(int w = 0; w < layer_ptr[l - 1]; ++w)
			{
				error_output_ptr[w] += gradient_ptr[n] * weight_ptr[w];
			}
		}
	}
}

void ghog::lib::MultilayerPerceptron::update_weights()
{
	int* layer_ptr = _layers.ptr< int >(0);
	layer_ptr++; //Skip input layer;
	for(int l = 0; l < _weights.size(); ++l)
	{
		float* input_ptr = _last_results.ptr< float >(l);
		float* gradient_ptr = _last_gradients.ptr< float >(l);
		float* bias_ptr = _biases.ptr< float >(l);
		for(int n = 0; n < layer_ptr[l]; ++n)
		{
			float* weight_ptr = _weights[l].ptr< float >(n);
			bias_ptr[n] -= _learning_rate * gradient_ptr[n];
			//Use number of nodes of previous layer as number of inputs.
			for(int w = 0; w < layer_ptr[l - 1]; ++w)
			{
				weight_ptr[w] -= gradient_ptr[n] * input_ptr[w]
					* _learning_rate;
			}
		}
	}
}

void ghog::lib::MultilayerPerceptron::train_multiple_times(cv::Mat inputs,
	cv::Mat expected_outputs,
	int num_times)
{
	for(int i = 0; i < num_times; ++i)
	{
		train(inputs, expected_outputs);
	}
}

float ghog::lib::MultilayerPerceptron::activation(float sum)
{
	return tanh(sum);
}

float ghog::lib::MultilayerPerceptron::activation_derivative(float sum)
{
	float sec_hyp = 1 / (cosh(sum));
	return (sec_hyp * sec_hyp);
}
