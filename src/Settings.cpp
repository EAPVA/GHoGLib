/*
 * Settings.cpp
 *
 *  Created on: May 12, 2015
 *      Author: marcelo
 */

#include <include/Settings.h>

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

//template < typename T >
//T Settings::load(std::string module,
//	std::string attribute)
//{
//	return _file.FirstChildElement(module.c_str())->FindAttribute(
//		attribute.c_str());
//}

template < >
int Settings::load(std::string module,
	std::string attribute)
{
	return _file.FirstChildElement(module.c_str())->IntAttribute(
		attribute.c_str());
}

template < >
float Settings::load(std::string module,
	std::string attribute)
{
	return _file.FirstChildElement(module.c_str())->FloatAttribute(
		attribute.c_str());
}

template < >
std::string Settings::load(std::string module,
	std::string attribute)
{
	return _file.FirstChildElement(module.c_str())->Attribute(attribute.c_str());
}

template < typename T >
void Settings::save(std::string module,
	std::string attribute,
	T value)
{
	_file.FirstChildElement(module.c_str())->SetAttribute(attribute.c_str(),
		value);
}

//template < >
//void Settings::save(std::string module,
//	std::string attribute,
//	int value)
//{
//
//}
//
//template < >
//void Settings::save(std::string module,
//	std::string attribute,
//	float value)
//{
//
//}
//
//template < >
//void Settings::save(std::string module,
//	std::string attribute,
//	double value)
//{
//
//}

void Settings::save_file()
{
	_file.SaveFile(_filename.c_str());
}

void Settings::save_default_settings(std::string filename)
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLElement* descriptor_module = doc.NewElement("Descriptor");
	descriptor_module->SetAttribute("GRID_SIZE_COLS", 1);
	descriptor_module->SetAttribute("GRID_SIZE_ROWS", 3);
	descriptor_module->SetAttribute("BLOCK_SIZE_COLS", 1);
	descriptor_module->SetAttribute("BLOCK_SIZE_ROWS", 1);
	descriptor_module->SetAttribute("BLOCK_STRIDE_COLS", 1);
	descriptor_module->SetAttribute("BLOCK_STRIDE_ROWS", 1);
	descriptor_module->SetAttribute("NUMBER_OF_BINS", 9);
	doc.InsertEndChild(descriptor_module);
	tinyxml2::XMLElement* classifier_module = doc.NewElement("Classifier");
	classifier_module->SetAttribute("TYPE", "MLP");
	classifier_module->SetAttribute("FILENAME", "mlp.xml");
	doc.InsertEndChild(classifier_module);
	doc.SaveFile(filename.c_str());
}

} /* namespace lib */
} /* namespace ghog */
