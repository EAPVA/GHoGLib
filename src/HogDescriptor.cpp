/*
 * HogCPU.cpp
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#include <include/HogDescriptor.h>

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

void HogDescriptor::alloc_buffer(cv::Size buffer_size,
	int type,
	cv::Mat& buffer)
{
	buffer.create(buffer_size.height, buffer_size.width, type);
	buffer.setTo(0);
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

void HogDescriptor::image_normalization_sync(cv::Mat& image)
{
	//TODO
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

void HogDescriptor::calc_gradient_sync(cv::Mat input_img,
	cv::Mat& magnitude,
	cv::Mat& phase)
{
	//Store dx temporarily on magnitude matrix
	cv::Sobel(input_img, magnitude, -1, 1, 0, 1);
	//Store dy temporarily on phase matrix
	cv::Sobel(input_img, phase, -1, 0, 1, 1);
	cv::cartToPolar(magnitude, phase, magnitude, phase, true);
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

void HogDescriptor::create_descriptor_sync(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat& descriptor)
{
	//TODO: verify that magnitude and phase have same size and type.
	//TODO: verify that the descriptor has correct size and type
	//TODO: possibly preallocate histograms auxiliary matrix

	cv::Size cell_grid = Utils::partition(magnitude.size(), _cell_size);
	cv::Mat histograms[cell_grid.height];
	cv::Size block_grid(
		((cell_grid.width - _block_size.width) / _block_stride.width) + 1,
		((cell_grid.height - _block_size.height) / _block_stride.height) + 1);
	int total_cells = cell_grid.width * cell_grid.height;
	int total_blocks = block_grid.width * block_grid.height;
	int cells_per_block = _block_size.width * _block_size.height;
	int total_histograms = total_blocks * cells_per_block;
	int total_outputs = total_histograms * _num_bins;
	cv::Size histograms_size(_num_bins, cell_grid.width);
	for(int i = 0; i < cell_grid.height; ++i)
	{
		alloc_buffer(histograms_size, CV_32FC1, histograms[i]);
	}
	int top_row = 0, bottom_row = 0, left_col = 0, right_col = 0;
	int extra_rows = magnitude.rows % cell_grid.height;
	int extra_cols = magnitude.cols % cell_grid.width;

	for(int i = 0; i < cell_grid.height; ++i)
	{
		bottom_row = top_row + _cell_size.height;
		if(extra_rows > 0)
		{
			extra_rows--;
			bottom_row++;
		}
		cv::Mat mag_aux = magnitude.rowRange(top_row, bottom_row);
		cv::Mat phase_aux = phase.rowRange(top_row, bottom_row);
		for(int j = 0; j < cell_grid.width; ++j)
		{
			right_col = left_col + _cell_size.width;
			if(extra_cols > 0)
			{
				extra_cols--;
				right_col++;
			}
			//Each element of the array histograms contains a ROW of the cell grid.
			//Each ROW of each element corresponds to a COLUMN of the cell grid.
			calc_histogram(mag_aux.colRange(left_col, right_col),
				phase_aux.colRange(left_col, right_col), histograms[i].row(j));
			left_col = right_col;
		}
		extra_cols = magnitude.cols % cell_grid.width;
		top_row = bottom_row;
	}

	left_col = 0;
	int col_stride = _num_bins * cells_per_block;
	right_col += col_stride;

	int block_posx = 0;
	int block_posy = 0;

	for(int i = 0; i < block_grid.height; ++i)
	{
		for(int j = 0; j < block_grid.width; ++j)
		{
			for(int k = 0; k < _block_size.height; ++k)
			{
				for(int l = 0; l < _block_size.width; ++l)
				{
					histograms[block_posy + k].row(block_posx + l).copyTo(
						descriptor.colRange(left_col, right_col));
					left_col = right_col;
					right_col += col_stride;
				}
			}
			block_posx += _block_stride.width;
		}
		block_posy += _block_stride.height;
		block_posx = 0;
	}

}

void HogDescriptor::calc_histogram(cv::Mat magnitude,
	cv::Mat phase,
	cv::Mat cell_histogram)
{
	float bin_size = 360.0f / (float)_num_bins;

	int left_bin, right_bin;
	float delta;

	float mag_total = 0.0f;

	for(int i = 0; i < magnitude.rows; ++i)
	{
		for(int j = 0; j < magnitude.cols; ++j)
		{
			if(magnitude.at< float >(i, j) > 0)
			{
				left_bin = (int)floor(
					(phase.at< float >(i, j) - bin_size / 2.0f) / bin_size);
				if(left_bin < 0)
					left_bin += _num_bins;
				right_bin = (left_bin + 1) % _num_bins;

				delta = (phase.at< float >(i, j) / bin_size) - right_bin;
				if(right_bin == 0)
					delta -= _num_bins;

				cell_histogram.at< float >(left_bin) += (0.5 - delta)
					* magnitude.at< float >(i, j);
				cell_histogram.at< float >(right_bin) += (0.5 + delta)
					* magnitude.at< float >(i, j);
				mag_total += magnitude.at< float >(i, j);
			}
		}
	}

	for(int i = 0; i < _num_bins; ++i)
	{
		cell_histogram.at< float >(i) /= mag_total;
	}
}

void HogDescriptor::normalize_blocks(cv::Mat& descriptor)
{
	int cells_per_block = _block_size.height * _block_size.width;
	int elements_per_block = cells_per_block * _num_bins;

	for(int i = 0; i < descriptor.cols; i += elements_per_block)
	{
		float L1_norm = 0.0f;
		for(int j = 0; j < elements_per_block; ++j)
		{
			L1_norm += descriptor.at< float >(i + j);
		}
		for(int j = 0; j < elements_per_block; ++j)
		{
			descriptor.at< float >(i + j) = sqrt(
				descriptor.at< float >(i + j) / L1_norm);
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
	_block_size.width = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_COLS");
	_block_size.height = _settings.load_int(std::string("Descriptor"),
		"BLOCK_SIZE_ROWS");
	_cell_size.width = _settings.load_int(std::string("Descriptor"),
		"CELL_SIZE_COLS");
	_cell_size.height = _settings.load_int(std::string("Descriptor"),
		"CELL_SIZE_ROWS");
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
		|| (param_name == "BLOCK_SIZE_ROWS")
		|| (param_name == "NUMBER_OF_BINS"))
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

} /* namespace lib */
} /* namespace ghog */
