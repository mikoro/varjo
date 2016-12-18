// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

#include "Utils/Settings.h"

using namespace Varjo;

namespace po = boost::program_options;

bool Settings::load(int argc, char** argv)
{
	po::options_description options("Options");

	options.add_options()

		("help", "")

		("general.cudaDeviceNumber", po::value(&general.cudaDeviceNumber)->default_value(0), "")
		("general.checkGLErrors", po::value(&general.checkGLErrors)->default_value(true), "")
		("general.filmScale", po::value(&general.filmScale)->default_value(0.5f), "")
		("general.infoPanelState", po::value(&general.infoPanelState)->default_value(2), "")
		("general.infoPanelFontSize", po::value(&general.infoPanelFontSize)->default_value(18), "")

		("window.width", po::value(&window.width)->default_value(1280), "")
		("window.height", po::value(&window.height)->default_value(800), "")
		("window.fullscreen", po::value(&window.fullscreen)->default_value(false), "")
		("window.vsync", po::value(&window.vsync)->default_value(false), "")
		("window.hideCursor", po::value(&window.hideCursor)->default_value(false), "")
		;
	
	std::ifstream iniFile("varjo.ini");
	po::variables_map vm;

	try
	{
		po::store(po::parse_command_line(argc, argv, options), vm);
		po::store(po::parse_config_file(iniFile, options), vm);
		po::notify(vm);
	}
	catch (const po::error& e)
	{
		std::cout << "Command line / settings file parsing failed: " << e.what() << std::endl;
		std::cout << "Try '--help' for list of valid options" << std::endl;

		return false;
	}

	if (vm.count("help"))
	{
		std::cout << options;
		return false;
	}

	return true;
}
