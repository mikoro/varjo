// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "App.h"
#include "Utils/Log.h"

using namespace Varjo;

int main(int argc, char** argv)
{
	int result = 0;

	try
	{
		result = App::getApp().run(argc, argv);
	}
	catch (...)
	{
		App::getLog().logException(std::current_exception());
		return -1;
	}

	cudaDeviceReset();
	return result;
}
