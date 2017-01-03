// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "tinyformat/tinyformat.h"

#include "Cuda/CudaUtils.h"
#include "Utils/App.h"
#include "Utils/StringUtils.h"

#include <cuda_gl_interop.h>

using namespace Varjo;

void CudaUtils::checkError(cudaError_t code, const std::string& message)
{
	if (code != cudaSuccess)
		throw std::runtime_error(tfm::format("CUDA error: %s: %s", message, cudaGetErrorString(code)));
}

void CudaUtils::initCuda()
{
	Log& log = App::getLog();
	Settings& settings = App::getSettings();

	log.logInfo("Initializing CUDA");

	int runtimeVersion;
	checkError(cudaRuntimeGetVersion(&runtimeVersion), "Could not get CUDA runtime version");

	int driverVersion;
	checkError(cudaDriverGetVersion(&driverVersion), "Could not get CUDA driver version");

	log.logInfo("CUDA Runtime version: %d | Driver version: %d", runtimeVersion, driverVersion);

	int deviceCount = 0;
	checkError(cudaGetDeviceCount(&deviceCount), "Could not get CUDA device count");

	unsigned int glDeviceCount = 4;
	int glDevices[4];
	checkError(cudaGLGetDevices(&glDeviceCount, glDevices, glDeviceCount, cudaGLDeviceListAll), "Could not get CUDA devices for current OpenGL context");

	log.logInfo("CUDA device count: %d | For current OpenGL context: %d", deviceCount, glDeviceCount);

	if (glDeviceCount < 1)
		throw std::runtime_error("Could not find any CUDA devices for current OpenGL context");

	int deviceNumber = glDevices[0];

	checkError(cudaSetDevice(deviceNumber), "Could not set CUDA device");

	cudaDeviceProp deviceProp;
	checkError(cudaGetDeviceProperties(&deviceProp, deviceNumber), "Could not get CUDA device properties");

	log.logInfo("CUDA selected device: %d | Name: %s | PCI domain/bus/device: %d/%d/%d", deviceNumber, deviceProp.name, deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
	log.logInfo("CUDA Compute capability: %d.%d | Total memory: %s", deviceProp.major, deviceProp.minor, StringUtils::humanizeNumber(double(deviceProp.totalGlobalMem), true));
}

void CudaUtils::calculateDimensions(const void* kernel, const char* name, uint32_t length, int& blockSize, int& gridSize)
{
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, length);

	gridSize = (length + blockSize - 1) / blockSize;

	App::getLog().logInfo("Kernel (%s) block size: %d | grid size: %d", name, blockSize, gridSize);
}

void CudaUtils::calculateDimensions2D(const void* kernel, const char* name, uint32_t width, uint32_t height, dim3& blockSize, dim3& gridSize)
{
	int tempBlockSize;
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &tempBlockSize, kernel, 0, width * height);

	blockSize.x = 32;
	blockSize.y = tempBlockSize / 32;

	if (blockSize.y == 0)
		blockSize.y = 1;

	gridSize.x = (width + blockSize.x - 1) / blockSize.x;
	gridSize.y = (height + blockSize.y - 1) / blockSize.y;

	App::getLog().logInfo("Kernel (%s) block size: %d (%dx%d) | grid size: %d (%dx%d)", name, tempBlockSize, blockSize.x, blockSize.y, gridSize.x * gridSize.y, gridSize.x, gridSize.y);
}
