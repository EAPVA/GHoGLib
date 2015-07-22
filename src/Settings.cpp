/*
 * Settings.cpp
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#include <include/Settings.h>

#include <include/GHogLibConstants.inc>

namespace ghog
{
namespace lib
{

Settings::Settings(std::string filename) :
	_filename(filename)
{
	while(_file.LoadFile(_filename.c_str()) != tinyxml2::XML_NO_ERROR)
	{
		Settings::save_default_settings(filename);
	}
}

Settings::~Settings()
{
	// TODO Auto-generated destructor stub
}

int Settings::load_int(std::string module,
	std::string attribute)
{
	return _file.FirstChildElement(module.c_str())->IntAttribute(
		attribute.c_str());
}

float Settings::load_float(std::string module,
	std::string attribute)
{
	return _file.FirstChildElement(module.c_str())->FloatAttribute(
		attribute.c_str());
}

std::string Settings::load_str(std::string module,
	std::string attribute)
{
	return _file.FirstChildElement(module.c_str())->Attribute(attribute.c_str());
}

void Settings::load_file(std::string filename)
{
	if(_file.LoadFile(_filename.c_str()) != tinyxml2::XML_NO_ERROR)
	{
		_filename = filename;
	} else
	{
		_file.LoadFile(_filename.c_str());
	}
}

void Settings::save_file()
{
	_file.SaveFile(_filename.c_str());
}

void Settings::save_default_settings(std::string filename)
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLElement* descriptor_module = doc.NewElement("Descriptor");
	descriptor_module->SetAttribute("NUMBER_OF_BINS",
		GHOG_LIB_DEFAULT_NUM_BINS);
	descriptor_module->SetAttribute("CELL_SIZE_COLS",
		GHOG_LIB_DEFAULT_CELL_SIZE.width);
	descriptor_module->SetAttribute("CELL_SIZE_ROWS",
		GHOG_LIB_DEFAULT_CELL_SIZE.height);
	descriptor_module->SetAttribute("BLOCK_SIZE_COLS",
		GHOG_LIB_DEFAULT_BLOCK_SIZE.width);
	descriptor_module->SetAttribute("BLOCK_SIZE_ROWS",
		GHOG_LIB_DEFAULT_BLOCK_SIZE.height);
	descriptor_module->SetAttribute("BLOCK_STRIDE_COLS",
		GHOG_LIB_DEFAULT_BLOCK_STRIDE.width);
	descriptor_module->SetAttribute("BLOCK_STRIDE_ROWS",
		GHOG_LIB_DEFAULT_BLOCK_STRIDE.height);
	descriptor_module->SetAttribute("DETECTION_WINDOW_COLS",
		GHOG_LIB_DEFAULT_WIN_SIZE.width);
	descriptor_module->SetAttribute("DETECTION_WINDOW_ROWS",
		GHOG_LIB_DEFAULT_WIN_SIZE.height);
	doc.InsertEndChild(descriptor_module);
	tinyxml2::XMLElement* classifier_module = doc.NewElement("Classifier");
	classifier_module->SetAttribute("TYPE", "MLP");
	classifier_module->SetAttribute("FILENAME", "mlp.xml");
	doc.InsertEndChild(classifier_module);
	doc.SaveFile(filename.c_str());
}

} /* namespace lib */
} /* namespace ghog */
