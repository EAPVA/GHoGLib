/*
 * Settings.h
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#ifndef SETTINGS_H_
#define SETTINGS_H_

#include <string>

#include <3rdParty/tinyxml2.h>

namespace ghog
{
namespace lib
{

class Settings
{
public:
	Settings(std::string filename);
	virtual ~Settings();

	int load_int(std::string module,
		std::string attribute);
	float load_float(std::string module,
		std::string attribute);
	std::string load_str(std::string module,
		std::string attribute);
	template < typename T >
	void save(std::string module,
		std::string attribute,
		T value);

	void load_file(std::string filename);
	void save_file();

protected:
	tinyxml2::XMLDocument _file;
	std::string _filename;

	void virtual save_default_settings(std::string filename);
};

} /* namespace lib */
} /* namespace ghog */
#endif /* SETTINGS_H_ */
