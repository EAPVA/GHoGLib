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

/**
 * \brief Extensions to the settings class to load and save MLP weights.
 */
class MLPSettings: public Settings
{
public:
	/**
	 * \brief Load from file constructor.
	 *
	 * Opens and loads the XML file specified. If the XML file can't be found,
	 * creates one using default parameters.
	 *
	 * \param filename Relative or absolute path to XML file.
	 */
	MLPSettings(std::string filename);
	/**
	 * \brief Default destructor.
	 */
	virtual ~MLPSettings();

	/**
	 * \brief Saves a single layer on a XML file.
	 *
	 * \warning Not yet implemented.
	 */
	void save_layers(cv::Mat layers);
	/**
	 * \brief Loads a single layer from a XML file.
	 *
	 * \warning Not implemented yet.
	 */
	cv::Mat load_layers();
	/**
	 * \brief Saves all weights to a XML file.
	 *
	 * \warning Not implemented yet.
	 */
	void save_weights(std::vector< cv::Mat > weights);
	/**
	 * \brief Loads all weights from a a file;
	 */
	std::vector< cv::Mat > load_weights();

protected:
	void virtual save_default_settings(std::string filename);

};

} /* namespace lib */
} /* namespace ghog */
#endif /* MLPSETTINGS_H_ */
