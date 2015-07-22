/*
 * Settings.h
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#ifndef GHOGLIB_SETTINGS_H_
#define GHOGLIB_SETTINGS_H_

#include <string>

#include <3rdParty/tinyxml2.h>

namespace ghog
{
namespace lib
{

/**
 * \brief Helper class to manipulate XML configuration files.
 */
class Settings
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
	Settings(std::string filename);
	/**
	 * Default destructor.
	 */
	virtual ~Settings();

	/**
	 * \brief Loads a parameter as an integer.
	 *
	 * \param module Name of the module where the parameter is.
	 * \param attribute Name of the parameter.
	 *
	 * \returns The value of the parameter, as an integer.
	 */
	int load_int(std::string module,
		std::string attribute);
	/**
	 * \brief Loads a parameter as a floating point number.
	 *
	 * \param module Name of the module where the parameter is.
	 * \param attribute Name of the parameter.
	 *
	 * \returns The value of the parameter, as a floating point number.
	 */
	float load_float(std::string module,
		std::string attribute);
	/**
	 * \brief Loads a parameter as a string.
	 *
	 * \param module Name of the module where the parameter is.
	 * \param attribute Name of the parameter.
	 *
	 * \returns The value of the parameter, as a string.
	 */
	std::string load_str(std::string module,
		std::string attribute);
	/**
	 * \brief Saves a parameter.
	 *
	 * \param module Name of the module where the parameter is.
	 * \param attribute Name of the parameter.
	 * \param value New value of the parameter.
	 */
	template< typename T >
	void save(std::string module,
		std::string attribute,
		T value);

	/**
	 * \brief Opens and loads the XML file specified. If the XML file can't be found,
	 * creates one using default parameters.
	 *
	 * \param filename Relative or absolute path to XML file.
	 */
	void load_file(std::string filename);
	/**
	 * \brief Saves the current open file.
	 *
	 * \bug Not saving the file correctly.
	 */
	void save_file();

	std::string _filename; /**<Name of the opened file.*/

protected:
	tinyxml2::XMLDocument _file; /**<Represents the opened XML file*/

	/**
	 * \brief Creates a XML file using the default GHogLib parameters.
	 *
	 * \param filename Path of the file being created.
	 */
	void virtual save_default_settings(std::string filename);
};

template< typename T >
void Settings::save(std::string module,
	std::string attribute,
	T value)
{
	_file.FirstChildElement(module.c_str())->SetAttribute(attribute.c_str(),
		value);
}

} /* namespace lib */
} /* namespace ghog */
#endif /* GHOGLIB_SETTINGS_H_ */
