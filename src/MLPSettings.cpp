/*
 * MLPSettings.cpp
 *
 *  Created on: May 26, 2015
 *      Author: marcelo
 */

#include <include/MLPSettings.h>

namespace ghog
{
namespace lib
{

MLPSettings::~MLPSettings()
{
	// TODO Auto-generated destructor stub
}

static void MLPSettings::save_default_settings(std::string filename)
{
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLElement* train_module = doc.NewElement("Training");
	train_module->SetAttribute("LEARNING_RATE", 0.2f);
	train_module->SetAttribute("MAX_ITERATIONS", 1000);
	train_module->SetAttribute("TARGET_ERROR", 10e-6);
	doc.InsertEndChild(train_module);
	doc.SaveFile(filename.c_str());
}

void MLPSettings::save_layers(cv::Mat layers)
{

}

cv::Mat MLPSettings::load_layers()
{

}

void MLPSettings::save_weights(std::vector< cv::Mat > weights)
{

}

std::vector< cv::Mat > MLPSettings::load_weights()
{

}

} /* namespace lib */
} /* namespace ghog */
