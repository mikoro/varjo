// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Utils/App.h"
#include "Utils/Log.h"

using namespace Varjo;

int run(int argc, char** argv)
{
	try
	{
		if (!App::getSettings().load(argc, argv))
			return -1;

		App::getWindow().run();
	}
	catch (...)
	{
		App::getLog().logException(std::current_exception());
		return -1;
	}

	return 0;
}

int main(int argc, char** argv)
{
	int result = run(argc, argv);
	cudaDeviceReset();
	return result;
}
