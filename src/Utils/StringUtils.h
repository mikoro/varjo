// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Varjo
{
	class StringUtils
	{
	public:

		static bool endsWith(const std::string& input, const std::string& end);
		static std::string readFileToString(const std::string& fileName);
		static std::string humanizeNumber(double value, bool usePowerOfTwo = false);
	};
}
