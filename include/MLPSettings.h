/*
 * MLPSettings.h
 *
 *  Created on: May 26, 2015
 *      Author: marcelo
 */

#ifndef MLPSETTINGS_H_
#define MLPSETTINGS_H_

#include "Settings.h"

#include <vector>

#include <opencv2/core/core.hpp>

namespace ghog
{
namespace lib
{

class MLPSettings: public ghog::lib::Settings
{
public:
	virtual ~MLPSettings();

	void save_layers(cv::Mat layers);
	cv::Mat load_layers();
	void save_weights(std::vector< cv::Mat > weights);
	std::vector< cv::Mat > load_weights();

protected:
	static void virtual save_default_settings(std::string filename);

};

} /* namespace lib */
} /* namespace ghog */
#endif /* MLPSETTINGS_H_ */
