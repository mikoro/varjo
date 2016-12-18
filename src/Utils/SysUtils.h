// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

namespace Varjo
{
	enum class ConsoleTextColor
	{
		DEFAULT,
		GRAY_ON_BLACK,
		WHITE_ON_BLACK,
		YELLOW_ON_BLACK,
		WHITE_ON_RED
	};

	class SysUtils
	{
	public:

		static void openFileExternally(const std::string& fileName);
		static void setConsoleTextColor(ConsoleTextColor color);
		static uint64_t getFileSize(const std::string& fileName);
		static std::vector<std::string> getAllFiles(const std::string& dirName);
	};
}
