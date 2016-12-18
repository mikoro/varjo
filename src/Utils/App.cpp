// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Utils/App.h"

using namespace Varjo;

Log& App::getLog()
{
	static Log log("varjo.log");
	return log;
}

Settings& App::getSettings()
{
	static Settings settings;
	return settings;
}

Window& App::getWindow()
{
	static Window window;
	return window;
}
