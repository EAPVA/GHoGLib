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

	template < typename T >
	T load(std::string module,
		std::string attribute);
	template < typename T >
	void save(std::string module,
		std::string attribute,
		T value);

	void save_file();

protected:
	tinyxml2::XMLDocument _file;
	std::string _filename;
};

} /* namespace lib */
} /* namespace ghog */
#endif /* SETTINGS_H_ */
