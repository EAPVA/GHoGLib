/*
 * HogCPU.cpp
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#include <include/HogDescriptor.h>

#include "math_constants.h"

#include <boost/thread.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#include <include/Utils.h>

namespace ghog
{
namespace lib
{

HogDescriptor::HogDescriptor(std::string settings_file) :
	_settings(settings_file)
{
	_classifier = NULL;

	load_settings(settings_file);
}

HogDescriptor::~HogDescriptor()
{
// TODO Auto-generated destructor stub
}

GHOG_LIB_STATUS HogDescriptor::alloc_buffer(cv::Size buffer_size,
	int type,
	cv::Mat& buffer,
	int padding_size)
{
	cv::Mat buffer_padding(buffer_size.height + 2 * padding_size,
		buffer_size.width + 2 * padding_size, type);
	buffer_padding.setTo(0);
	buffer = buffer_padding.rowRange(padding_size,
		buffer_padding.rows - padding_size).colRange(padding_size,
		buffer_padding.cols - padding_size);
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS HogDescriptor::image_normalization(cv::Mat& image,
	ImageCallback* callback)
{
	boost::thread(&HogDescriptor::image_normalization_async, this, image,
		callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void HogDescriptor::image_normalization_async(cv::Mat& image,
	ImageCallback* callback)
{
	image_normalization_sync(image);
	callback->image_processed(image);
}

GHOG_LIB_STATUS HogDescriptor::image_normalization_sync(cv::Mat& image)
{
	for(int i = 0; i < image.rows; ++i)
	{
		float* input_ptr = image.ptr< float >(i);
		for(int j = 0; j < image.cols; ++j)
		{
			for(int k = 0; k < 3; ++k)
			{
				input_ptr[3 * j + k] = sqrt(input_ptr[3 * j + k] / 256.0);
			}
		}
	}
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS HogDescriptor::calc_gradient(cv::Mat input_img,
	cv::Mat& magnitude,
	cv::Mat& phase,
	GradientCallback* callback)
{
	boost::thread(&HogDescriptor::calc_gradient_async, this, input_img,
		magnitude, phase, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void HogDescriptor::calc_gradient_async(cv::Mat input_img,
	cv::Mat& magnitude,
	cv::Mat& phase,
	GradientCallback* callback)
{
	calc_gradient_sync(input_img, magnitude, phase);
	callback->gradients_obtained(magnitude, phase);
}

GHOG_LIB_STATUS HogDescriptor::calc_gradient_sync(cv::Mat input_img,
	cv::Mat& magnitude,
	cv::Mat& phase)
{
	for(int i = 0; i < input_img.rows; ++i)
	{
		float* input_ptr = input_img.ptr< float >(i);
		float* magnitude_ptr = magnitude.ptr< float >(i);
		float* phase_ptr = phase.ptr< float >(i);
		for(int j = 0; j < input_img.cols; ++j)
		{
			float mag_max = 0.0f;
			float phase_max = 0.0f;
			float dx, dy;
			for(int k = 0; k < 3; ++k)
			{
				dx = input_ptr[3 * j + k + 3] - input_ptr[3 * j + k - 3];
				dy = input_ptr[3 * j + k + input_img.step1()]
					- input_ptr[3 * j + k - input_img.step1()];
				if(i == 0 || i == input_img.rows - 1)
				{
					dy = 0;
				}

				if(j == 0 || j == input_img.cols - 1)
				{
					dx = 0;
				}

				float mag = sqrt(dx * dx + dy * dy);
				if(mag > mag_max)
				{
					mag_max = mag;
					phase_max = (atan(dy / dx) / CUDART_PI_F) + 0.5f;
				}
			}
			magnitude_ptr[j] = mag_max;
			phase_ptr[j] = phase_max;
		}
	}
	return GHOG_LIB_STATUS_OK;
}

GHOG_LIB_STATUS HogDescriptor::create_descriptor(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat& descriptor,
	DescriptorCallback* callback)
{
	boost::thread(&HogDescriptor::create_descriptor_async, this, magnitude,
		phase, descriptor, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void HogDescriptor::create_descriptor_async(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat& descriptor,
	DescriptorCallback* callback)
{
	create_descriptor_sync(magnitude, phase, descriptor);
	callback->descriptor_obtained(descriptor);
}

GHOG_LIB_STATUS HogDescriptor::create_descriptor_sync(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat& descriptor)
{
	cv::Mat histograms;
	cv::Size histograms_size(_cell_grid.width, _cell_grid.height);
	alloc_buffer(histograms_size, CV_32FC(9), histograms, 0);
	return create_descriptor_sync(magnitude, phase, descriptor, histograms);
}

GHOG_LIB_STATUS HogDescriptor::create_descriptor_sync(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat& descriptor,
	cv::Mat& histograms)
{
	if((magnitude.size() != _window_size) || (phase.size() != _window_size))
	{
		std::cout << "Erro ao chamar a função create_descriptor_sync. "
			<< "Imagens de entrada tem tamanho diferente do esperado"
			<< std::endl;
		return GHOG_LIB_STATUS_INVALID_IMAGE_SIZE;
	}
//TODO: verify that magnitude and phase have correct type.
//TODO: verify that the descriptor has correct size and type

	cv::Size block_grid(
		((_cell_grid.width - _block_size.width) / _block_stride.width) + 1,
		((_cell_grid.height - _block_size.height) / _block_stride.height) + 1);
	int cells_per_block = _block_size.width * _block_size.height;
	cv::Size histograms_size(_num_bins, _cell_grid.width);
	int top_row = 0, bottom_row = 0, left_col = 0, right_col = 0;

	for(int i = 0; i < _cell_grid.height; ++i)
	{
		bottom_row = top_row + _cell_size.height;
		cv::Mat mag_aux = magnitude.rowRange(top_row, bottom_row);
		cv::Mat phase_aux = phase.rowRange(top_row, bottom_row);
		for(int j = 0; j < _cell_grid.width; ++j)
		{
			right_col = left_col + _cell_size.width;
			//Each element of the array histograms contains a histogram of one cell.
			calc_histogram(mag_aux.colRange(left_col, right_col),
				phase_aux.colRange(left_col, right_col),
				histograms.row(i).col(j));
			left_col = right_col;
		}
		left_col = 0;
		top_row = bottom_row;
	}

	left_col = 0;
	right_col = _num_bins;

	int block_posx = 0;
	int block_posy = 0;

	for(int i = 0; i < block_grid.height; ++i)
	{
		for(int j = 0; j < block_grid.width; ++j)
		{
			for(int k = 0; k < _block_size.height; ++k)
			{
				float* histograms_ptr = histograms.ptr< float >(block_posy + k);
				for(int l = 0; l < _block_size.width; ++l)
				{
					cv::Mat descriptor_aux = descriptor.colRange(left_col,
						right_col);
					for(int m = 0; m < _num_bins; ++m)
					{
						descriptor_aux.at< float >(m) = histograms_ptr[m];
					}
					left_col = right_col;
					right_col = left_col + _num_bins;
					histograms_ptr += _num_bins;
				}
			}
			block_posx += _block_stride.width;
		}
		block_posy += _block_stride.height;
		block_posx = 0;
	}
	normalize_blocks(descriptor);
	return GHOG_LIB_STATUS_OK;
}

void HogDescriptor::calc_histogram(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat cell_histogram)
{
	float bin_size = 1.0f / (float)_num_bins;

	int left_bin, right_bin;
	float delta;

	float mag_total = 0.0f;
	float* cell_histogram_ptr = cell_histogram.ptr< float >(0);

	for(int i = 0; i < _num_bins; ++i)
	{
		cell_histogram_ptr[i] = 0.0f;
	}

	for(int i = 0; i < magnitude.rows; ++i)
	{
		for(int j = 0; j < magnitude.cols; ++j)
		{
			if(magnitude.at< float >(i, j) > 0)
			{
				left_bin = (int)floor(
					(phase.at< float >(i, j) - bin_size / 2.0f) / bin_size);
				left_bin = (left_bin + _num_bins) % _num_bins;
				right_bin = (left_bin + 1);

				//Might be outside the range. First use on the formula below, then fix the range.
				delta = (phase.at< float >(i, j) / bin_size) - right_bin;
				if(delta < -0.5)
				{
					delta += _num_bins;
				}

				//Fix range for right_bin
				right_bin = right_bin % _num_bins;

				cell_histogram_ptr[left_bin] += (0.5 - delta)
					* magnitude.at< float >(i, j);
				cell_histogram_ptr[right_bin] += (0.5 + delta)
					* magnitude.at< float >(i, j);
				mag_total += magnitude.at< float >(i, j);
			}
		}
	}
}

void HogDescriptor::normalize_blocks(cv::Mat& descriptor)
{
	int cells_per_block = _block_size.height * _block_size.width;
	int elements_per_block = cells_per_block * _num_bins;

	for(int i = 0; i < descriptor.cols; i += elements_per_block)
	{
		float L_norm = 0.0f;
		for(int j = 0; j < elements_per_block; ++j)
		{
			float aux = descriptor.at< float >(i + j);
			switch(_norm_type)
			{
			case GHOG_LIB_NORM_TYPE_L1_SQRT:
				L_norm += aux;
			break;
			case GHOG_LIB_NORM_TYPE_L2_HYS:
				L_norm += aux * aux;
			break;
			default:
				return;
			}
		}
		L_norm += 0.00001;
		if(_norm_type == GHOG_LIB_NORM_TYPE_L2_HYS)
		{
			L_norm = sqrt(L_norm);
		}
		float new_mag = 0.0f;
		for(int j = 0; j < elements_per_block; ++j)
		{
			switch(_norm_type)
			{
			case GHOG_LIB_NORM_TYPE_L1_SQRT:
				descriptor.at< float >(i + j) = sqrt(
					descriptor.at< float >(i + j) / L_norm);
			break;
			case GHOG_LIB_NORM_TYPE_L2_HYS:
				float aux = descriptor.at< float >(i + j);
				aux /= L_norm;
				if(aux > 0.2)
				{
					aux = 0.2;
				}
				new_mag += aux * aux;
				descriptor.at< float >(i + j) = aux;
			}
		}
		new_mag = sqrt(new_mag);
		if(_norm_type == GHOG_LIB_NORM_TYPE_L2_HYS)
		{
			if(new_mag > 0)
			{
				for(int j = 0; j < elements_per_block; ++j)
				{
					descriptor.at< float >(i + j) /= new_mag;
				}
			}
		}

	}
}

GHOG_LIB_STATUS HogDescriptor::classify(cv::Mat img,
	ClassifyCallback* callback)
{
	boost::thread(&HogDescriptor::classify_async, this, img, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void HogDescriptor::classify_async(cv::Mat img,
	ClassifyCallback* callback)
{
	bool ret = classify_sync(img);
	callback->classification_result(ret);
}

bool HogDescriptor::classify_sync(cv::Mat img)
{
	bool ret = false;
	image_normalization_sync(img);
	cv::Mat grad_mag(img.size(), img.type());
	cv::Mat grad_phase(img.size(), img.type());
	calc_gradient_sync(img, grad_mag, grad_phase);
	cv::Mat descriptor;
	create_descriptor_sync(grad_mag, grad_phase, descriptor);
	cv::Mat output = _classifier->classify_sync(descriptor);
	if(output.at< float >(0) > 0)
	{
		ret = true;
	}
	return ret;
}

GHOG_LIB_STATUS HogDescriptor::locate(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride,
	LocateCallback* callback)
{
	boost::thread(&HogDescriptor::locate_async, this, img, roi, window_size,
		window_stride, callback).detach();
	return GHOG_LIB_STATUS_OK;
}

void HogDescriptor::locate_async(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride,
	LocateCallback* callback)
{
	std::vector< cv::Rect > ret = locate_sync(img, roi, window_size,
		window_stride);
	callback->objects_located(ret);
}

std::vector< cv::Rect > HogDescriptor::locate_sync(cv::Mat img,
	cv::Rect roi,
	cv::Size window_size,
	cv::Size window_stride)
{
	std::vector< cv::Rect > ret;
	return ret;
}

void HogDescriptor::load_settings(std::string filename)
{
	_num_bins = _settings.load_int(std::string("Descriptor"), "NUMBER_OF_BINS");
	_cell_size.width = _settings.load_int(std::string("Descriptor"),
		"CELL_SIZE_COLS");
	_cell_size.height = _settings.load_int(std::string("Descriptor"),
		"CELL_SIZE_ROWS");
	_block_size.width = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_COLS");
	_block_size.height = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_ROWS");
	_block_stride.width = _settings.load_int(std::string("Descriptor"),
		"BLOCK_STRIDE_COLS");
	_block_stride.height = _settings.load_int(std::string("Descriptor"),
		"BLOCK_STRIDE_ROWS");
	_window_size.height = _settings.load_int(std::string("Descriptor"),
		"DETECTION_WINDOW_ROWS");
	_window_size.width = _settings.load_int(std::string("Descriptor"),
		"DETECTION_WINDOW_COLS");

	_cell_grid = Utils::partition(_window_size, _cell_size);

	_norm_type = GHOG_LIB_NORM_TYPE_L2_HYS;
	cv::Size block_dim(_block_size.width * _cell_size.width,
		_block_size.height * _cell_size.height);
}

void HogDescriptor::set_classifier(IClassifier* classifier)
{
	_classifier = classifier;
}

GHOG_LIB_STATUS HogDescriptor::set_param(std::string param,
	std::string value)
{
	std::string module = get_module(param);
	if(module == "NULL")
	{
		return GHOG_LIB_STATUS_INVALID_PARAMETER_NAME;
	}
	_settings.save(module, param, value.c_str());
	load_settings(_settings._filename); //Ugly, change ASAP.
	return GHOG_LIB_STATUS_OK;
}

std::string HogDescriptor::get_param(std::string param)
{
	std::string module = get_module(param);
	if(module == "NULL")
	{
		return "Invalid parameter name.";
	} else
	{
		return _settings.load_str(module, param);
	}
}

std::string HogDescriptor::get_module(std::string param_name)
{
	if((param_name == "CELL_SIZE_COLS") || (param_name == "CELL_SIZE_ROWS")
		|| (param_name == "BLOCK_SIZE_COLS")
		|| (param_name == "BLOCK_SIZE_ROWS") || (param_name == "NUMBER_OF_BINS")
		|| (param_name == "BLOCK_STRIDE_COLS")
		|| (param_name == "BLOCK_STRIDE_ROWS")
//		|| (param_name == "CELL_GRID_COLS") || (param_name == "CELL_GRID_ROWS")
		|| (param_name == "DETECTION_WINDOW_COLS")
		|| (param_name == "DETECTION_WINDOWS_ROWS"))
	{
		return "Descriptor";
	} else if((param_name == "TYPE") || (param_name == "FILENAME"))
	{
		return "Classifier";
	} else
	{
		return "NULL";
	}
}

int HogDescriptor::get_descriptor_size()
{
	cv::Size block_grid(
		((_cell_grid.width - _block_size.width) / _block_stride.width) + 1,
		((_cell_grid.height - _block_size.height) / _block_stride.height) + 1);
	return block_grid.height * block_grid.width * _block_size.height
		* _block_size.width * _num_bins;
}

} /* namespace lib */
} /* namespace ghog */

