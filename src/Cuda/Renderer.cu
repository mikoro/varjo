// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Utils/App.h"
#include "Utils/CudaUtils.h"

#include "Cuda/Renderer.h"
#include "Cuda/Camera.h"
#include "Cuda/Filtering.h"
#include "Cuda/Intersect.h"
#include "Cuda/Kernels.h"
#include "Cuda/Material.h"
#include "Cuda/Misc.h"
#include "Cuda/Math.h"
#include "Cuda/Random.h"
#include "Cuda/Sampling.h"
#include "Cuda/Structs.h"

using namespace Varjo;

void Renderer::initialize(const Scene& scene)
{
	CameraData cameraData = scene.camera.getCameraData();

	CudaUtils::checkError(cudaMallocManaged(&camera, sizeof(CameraData)), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&nodes, sizeof(BVHNode) * scene.nodes.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&triangles, sizeof(Triangle) * scene.triangles.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&emitters, sizeof(uint32_t) * scene.emitters.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&materials, sizeof(Material) * scene.materials.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths, sizeof(Paths)), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->random, sizeof(Random) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->filmSample, sizeof(Sample) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->filmSamplePosition, sizeof(float2) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->throughput, sizeof(float3) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->result, sizeof(float3) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->length, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->extensionRay, sizeof(Ray) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->extensionIntersection, sizeof(Intersection) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->extensionBrdf, sizeof(float3) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->extensionBrdfPdf, sizeof(float) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->extensionCosine, sizeof(float) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->lightRay, sizeof(Ray) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->lightEmittance, sizeof(float3) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->lightBrdf, sizeof(float3) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->lightBrdfPdf, sizeof(float) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->lightPdf, sizeof(float) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->lightCosine, sizeof(float) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->lightRayBlocked, sizeof(bool) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues, sizeof(Queues)), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues->newPathQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues->diffuseMaterialQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues->extensionRayQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues->lightRayQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");

	memcpy(camera, &cameraData, sizeof(CameraData));
	memcpy(nodes, scene.nodes.data(), sizeof(BVHNode) * scene.nodes.size());
	memcpy(triangles, scene.triangles.data(), sizeof(Triangle) * scene.triangles.size());
	memcpy(emitters, scene.emitters.data(), sizeof(uint32_t) * scene.emitters.size());
	memcpy(materials, scene.materials.data(), sizeof(Material) * scene.materials.size());

	calculateDimensions(reinterpret_cast<void*>(initPathsKernel), "initPathsKernel", pathCount, initPathsBlockSize, initPathsGridSize);
	calculateDimensions(reinterpret_cast<void*>(clearPathsKernel), "clearPathsKernel", pathCount, clearPathsBlockSize, clearPathsGridSize);
	calculateDimensions(reinterpret_cast<void*>(logicKernel), "logicKernel", pathCount, logicBlockSize, logicGridSize);
	calculateDimensions(reinterpret_cast<void*>(newPathKernel), "newPathKernel", pathCount, newPathBlockSize, newPathGridSize);
	calculateDimensions(reinterpret_cast<void*>(diffuseMaterialKernel), "diffuseMaterialKernel", pathCount, diffuseMaterialBlockSize, diffuseMaterialGridSize);
	calculateDimensions(reinterpret_cast<void*>(extensionRayKernel), "extensionRayKernel", pathCount, extensionRayBlockSize, extensionRayGridSize);
	calculateDimensions(reinterpret_cast<void*>(lightRayKernel), "lightRayKernel", pathCount, lightRayBlockSize, lightRayGridSize);

	uint64_t time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
	initPathsKernel<<<initPathsGridSize, initPathsBlockSize>>>(paths, time, pathCount);
	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel (initPaths)");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel (initPaths)");

	averagePathsPerSecond.setAlpha(0.05f);
	averageRaysPerSecond.setAlpha(0.05f);
	emitterCount = uint32_t(scene.emitters.size());

	queues->newPathQueueLength = 0;
	queues->diffuseMaterialQueueLength = 0;
	queues->extensionRayQueueLength = 0;
	queues->lightRayQueueLength = 0;
}

void Renderer::shutdown()
{
	CudaUtils::checkError(cudaFree(camera), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(nodes), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(triangles), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(emitters), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(materials), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->random), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->filmSample), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->filmSamplePosition), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->throughput), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->result), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->length), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->extensionRay), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->extensionIntersection), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->extensionBrdf), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->extensionBrdfPdf), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->extensionCosine), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->lightRay), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->lightEmittance), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->lightBrdf), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->lightBrdfPdf), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->lightPdf), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->lightCosine), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->lightRayBlocked), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues->newPathQueue), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues->diffuseMaterialQueue), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues->extensionRayQueue), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues->lightRayQueue), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues), "Could not free CUDA device memory");
}

void Renderer::update(const Scene& scene)
{
	CameraData cameraData = scene.camera.getCameraData();
	memcpy(camera, &cameraData, sizeof(CameraData));
}

void Renderer::filmResized(uint32_t filmWidth, uint32_t filmHeight)
{
	if (pixels != nullptr)
		CudaUtils::checkError(cudaFree(pixels), "Could not free CUDA device memory");

	pixelCount = filmWidth * filmHeight;
	CudaUtils::checkError(cudaMallocManaged(&pixels, sizeof(Pixel) * pixelCount), "Could not allocate CUDA device memory");
	calculateDimensions(reinterpret_cast<void*>(clearPixelsKernel), "clearPixelsKernel", pixelCount, clearPixelsBlockSize, clearPixelsGridSize);
	calculateDimensions(reinterpret_cast<void*>(writeFilmKernel), "writeFilmKernel", pixelCount, writeFilmBlockSize, writeFilmGridSize);

	clear();
}

void Renderer::clear()
{
	clearPathsKernel<<<clearPathsGridSize, clearPathsBlockSize>>>(paths, pathCount);
	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel (clearPaths)");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel (clearPaths)");

	clearPixelsKernel<<<clearPixelsGridSize, clearPixelsBlockSize>>>(pixels, pixelCount);
	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel (clearPixels)");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel (clearPixels)");
}

void Renderer::render()
{
	Film& film = App::getWindow().getFilm();

	logicKernel<<<logicGridSize, logicBlockSize>>>(paths, queues, triangles, emitters, materials, pixels, pathCount, emitterCount, film.getWidth(), film.getHeight());
	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel (writePixels)");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel (writePixels)");

	if (queues->newPathQueueLength > 0)
	{
		newPathGridSize = (queues->newPathQueueLength + newPathBlockSize - 1) / newPathBlockSize;
		newPathKernel<<<newPathGridSize, newPathBlockSize>>>(paths, queues, camera, film.getWidth(), film.getHeight(), film.getLength());
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel (newPath)");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel (newPath)");
	}

	if (queues->diffuseMaterialQueueLength > 0)
	{
		diffuseMaterialGridSize = (queues->diffuseMaterialQueueLength + diffuseMaterialBlockSize - 1) / diffuseMaterialBlockSize;
		diffuseMaterialKernel<<<diffuseMaterialGridSize, diffuseMaterialBlockSize>>>(paths, queues, materials);
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel (diffuseMaterial)");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel (diffuseMaterial)");
	}

	if (queues->extensionRayQueueLength > 0)
	{
		extensionRayGridSize = (queues->extensionRayQueueLength + extensionRayBlockSize - 1) / extensionRayBlockSize;
		extensionRayKernel<<<extensionRayGridSize, extensionRayBlockSize>>>(paths, queues, nodes, triangles);
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel (extensionRay)");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel (extensionRay)");
	}

	if (queues->lightRayQueueLength > 0)
	{
		lightRayGridSize = (queues->lightRayQueueLength + lightRayBlockSize - 1) / lightRayBlockSize;
		lightRayKernel<<<lightRayGridSize, lightRayBlockSize>>>(paths, queues, nodes, triangles);
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel (lightRay)");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel (lightRay)");
	}

	cudaSurfaceObject_t filmSurfaceObject = film.getFilmSurfaceObject();
	writeFilmKernel<<<writeFilmGridSize, writeFilmBlockSize>>>(pixels, pixelCount, filmSurfaceObject, film.getWidth());
	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel (writeFilm)");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel (writeFilm)");
	film.releaseFilmSurfaceObject();

	float elapsedSeconds = timer.getElapsedSeconds();
	averagePathsPerSecond.addMeasurement(float(queues->newPathQueueLength) / elapsedSeconds);
	averageRaysPerSecond.addMeasurement(float(queues->extensionRayQueueLength + queues->lightRayQueueLength) / elapsedSeconds);
	timer.restart();

	queues->newPathQueueLength = 0;
	queues->diffuseMaterialQueueLength = 0;
	queues->extensionRayQueueLength = 0;
	queues->lightRayQueueLength = 0;
}

float Renderer::getPathsPerSecond() const
{
	return averagePathsPerSecond.getAverage();
}

float Renderer::getRaysPerSecond() const
{
	return averageRaysPerSecond.getAverage();
}
