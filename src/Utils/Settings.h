// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Varjo
{
	class Settings
	{
	public:

		bool load(int argc, char** argv);

		struct General
		{
			uint32_t cudaDeviceNumber;
		} general;

		struct Window
		{
			uint32_t width;
			uint32_t height;
			bool fullscreen;
			bool vsync;
			bool hideCursor;
			float renderScale;
			bool checkGLErrors;
		} window;
	};
}
