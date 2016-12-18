// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Utils/App.h"
#include "Utils/Log.h"
#include "Utils/CudaUtils.h"

using namespace Varjo;

int run(int argc, char** argv)
{
	Log& log = App::getLog();
	Settings& settings = App::getSettings();

	try
	{
		if (!settings.load(argc, argv))
			return -1;

		int deviceCount;
		CudaUtils::checkError(cudaGetDeviceCount(&deviceCount), "Could not get CUDA device count");
		CudaUtils::checkError(cudaSetDevice(settings.general.cudaDeviceNumber), "Could not set CUDA device");

		log.logInfo("CUDA selected device: %d (device count: %d)", settings.general.cudaDeviceNumber, deviceCount);

		cudaDeviceProp deviceProp;
		CudaUtils::checkError(cudaGetDeviceProperties(&deviceProp, settings.general.cudaDeviceNumber), "Could not get CUDA device properties");

		int driverVersion;
		CudaUtils::checkError(cudaDriverGetVersion(&driverVersion), "Could not get CUDA driver version");

		int runtimeVersion;
		CudaUtils::checkError(cudaRuntimeGetVersion(&runtimeVersion), "Could not get CUDA runtime version");

		log.logInfo("CUDA device: %s | Compute capability: %d.%d | Driver version: %d | Runtime version: %d", deviceProp.name, deviceProp.major, deviceProp.minor, driverVersion, runtimeVersion);

		App::getWindow().run();
	}
	catch (...)
	{
		log.logException(std::current_exception());
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
