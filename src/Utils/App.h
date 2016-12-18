// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Utils/Log.h"
#include "Utils/Settings.h"
#include "Utils/Window.h"

namespace Varjo
{
	class App
	{
	public:

		static Log& getLog();
		static Settings& getSettings();
		static Window& getWindow();
	};
}
