// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <vector>
#include <fstream>

#include "tinyformat/tinyformat.h"

#include "Utils/StringUtils.h"

using namespace Varjo;

bool StringUtils::endsWith(const std::string& input, const std::string& end)
{
	return input.rfind(end) == (input.size() - end.size());
}

std::string StringUtils::readFileToString(const std::string& fileName)
{
	std::ifstream file(fileName, std::ios::in | std::ios::binary | std::ios::ate);

	if (!file.good())
		throw std::runtime_error(tfm::format("Could not open file: %s", fileName));

	auto size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	file.read(&buffer[0], size);

	return std::string(&buffer[0], size);
}

std::string StringUtils::humanizeNumber(double value, bool usePowerOfTwo)
{
    const char* postfixes[] = { "", " k", " M", " G", " T", " P", " E", " Z", " Y" };
	const double divider = usePowerOfTwo ? 1024.0 : 1000.0;

    for (auto& postfix : postfixes)
	{
		if (value < divider)
            return tfm::format("%.2f%s", value, postfix);
		else
			value /= divider;
	}

    return tfm::format("%.2f%s", value, " Y");
}
