// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cstdint>

#include <cuda/helper_math.h>
#include <device_launch_parameters.h>

#include "Cuda/Renderer.h"
#include "Cuda/CudaUtils.h"
#include "Utils/App.h"
#include "Core/Ray.h"
#include "Core/Intersection.h"

using namespace Varjo;

namespace
{
	__device__ void initRay(Ray& ray)
	{
		ray.invD = make_float3(1.0, 1.0f, 1.0f) / ray.direction;
		ray.OoD = ray.origin / ray.direction;
	}

	__device__ Ray getRay(float2 pointOnFilm, const CameraData& camera)
	{
		float dx = pointOnFilm.x - camera.halfFilmWidth;
		float dy = pointOnFilm.y - camera.halfFilmHeight;

		float3 positionOnFilm = camera.filmCenter + (dx * camera.right) + (dy * camera.up);

		Ray ray;
		ray.origin = camera.position;
		ray.direction = normalize(positionOnFilm - camera.position);
		initRay(ray);

		return ray;
	}

	__device__ void intersectTriangle(const Triangle& triangle, const Ray& ray, Intersection& intersection)
	{
		bool hit = true;

		float3 v0v1 = triangle.vertices[1] - triangle.vertices[0];
		float3 v0v2 = triangle.vertices[2] - triangle.vertices[0];

		float3 pvec = cross(ray.direction, v0v2);
		float determinant = dot(v0v1, pvec);

		if (determinant == 0.0f)
			hit = false;

		float invDeterminant = 1.0f / determinant;

		float3 tvec = ray.origin - triangle.vertices[0];
		float u = dot(tvec, pvec) * invDeterminant;

		if (u < 0.0f || u > 1.0f)
			hit = false;

		float3 qvec = cross(tvec, v0v1);
		float v = dot(ray.direction, qvec) * invDeterminant;

		if (v < 0.0f || (u + v) > 1.0f)
			hit = false;

		float distance = dot(v0v2, qvec) * invDeterminant;

		if (distance < 0.0f || distance < ray.minDistance || distance > ray.maxDistance || distance > intersection.distance)
			hit = false;

		float w = 1.0f - u - v;

		if (hit)
		{
			intersection.wasFound = true;
			intersection.distance = distance;
			intersection.position = ray.origin + (distance * ray.direction);
			intersection.normal = w * triangle.normals[0] + u * triangle.normals[1] + v * triangle.normals[2];
			//intersection.texcoord = w * triangle.texcoords[0] + u * triangle.texcoords[1] + v * triangle.texcoords[2];
			intersection.materialIndex = triangle.materialIndex;
		}
	}

	__device__ bool intersectAabb(const AABB& aabb, const Ray& ray)
	{
		float x0 = fmaf(aabb.min.x, ray.invD.x, -ray.OoD.x);
		float y0 = fmaf(aabb.min.y, ray.invD.y, -ray.OoD.y);
		float z0 = fmaf(aabb.min.z, ray.invD.z, -ray.OoD.z);
		float x1 = fmaf(aabb.max.x, ray.invD.x, -ray.OoD.x);
		float y1 = fmaf(aabb.max.y, ray.invD.y, -ray.OoD.y);
		float z1 = fmaf(aabb.max.z, ray.invD.z, -ray.OoD.z);

		float tminbox = fmaxf(fmaxf(ray.minDistance, fminf(x0, x1)), fmaxf(fminf(y0, y1), fminf(z0, z1)));
		float tmaxbox = fminf(fminf(ray.maxDistance, fmaxf(x0, x1)), fminf(fmaxf(y0, y1), fmaxf(z0, z1)));
		
		return (tminbox <= tmaxbox);
	}

	__device__ void intersectBvh(const BVHNode* __restrict nodes, const Triangle* __restrict triangles, const Ray& ray, Intersection& intersection)
	{
		uint32_t stack[64];
		uint32_t stackIndex = 1;
		stack[0] = 0;

		BVHNode node;

		while (stackIndex > 0)
		{
			uint32_t nodeIndex = stack[--stackIndex];
			node = nodes[nodeIndex];

			// leaf node
			if (node.rightOffset == 0)
			{
				for (int i = 0; i < node.triangleCount; ++i)
					intersectTriangle(triangles[node.triangleOffset + i], ray, intersection);

				continue;
			}

			if (intersectAabb(node.aabb, ray))
			{
				stack[stackIndex++] = nodeIndex + 1; // left child
				stack[stackIndex++] = nodeIndex + uint32_t(node.rightOffset); // right child
			}
		}
	}

	__global__ void traceKernel(const CameraData* __restrict camera,
		const BVHNode* __restrict nodes,
		const Triangle* __restrict triangles,
		const Material* __restrict materials,
		cudaSurfaceObject_t film, uint32_t filmWidth, uint32_t filmHeight)
	{
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

		Ray ray = getRay(make_float2(x, y), *camera);
		Intersection intersection;
		float3 color = make_float3(0.0f, 0.0f, 0.0f);

		intersectBvh(nodes, triangles, ray, intersection);

		if (intersection.wasFound)
			color = materials[intersection.materialIndex].baseColor * dot(ray.direction, -intersection.normal);

		surf2Dwrite(make_float4(color, 1.0f), film, x * sizeof(float4), y, cudaBoundaryModeZero);
	}
}

void Renderer::initialize(const Scene& scene)
{
	CudaUtils::checkError(cudaMallocManaged(&camera, sizeof(CameraData)), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&nodes, sizeof(BVHNode) * scene.nodes.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&triangles, sizeof(Triangle) * scene.triangles.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&materials, sizeof(Material) * scene.materials.size()), "Could not allocate CUDA device memory");
	
	CameraData cameraData = scene.camera.getCameraData();

	memcpy(camera, &cameraData, sizeof(CameraData));
	memcpy(nodes, scene.nodes.data(), sizeof(BVHNode) * scene.nodes.size());
	memcpy(triangles, scene.triangles.data(), sizeof(Triangle) * scene.triangles.size());
	memcpy(materials, scene.materials.data(), sizeof(Material) * scene.materials.size());
}

void Renderer::shutdown()
{
	CudaUtils::checkError(cudaFree(camera), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(nodes), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(triangles), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(materials), "Could not free CUDA device memory");
}

void Renderer::update(const Scene& scene)
{
	CameraData cameraData = scene.camera.getCameraData();
	memcpy(camera, &cameraData, sizeof(CameraData));
}

void Renderer::filmResized(uint32_t width, uint32_t height)
{
	CudaUtils::calculateDimensions(static_cast<void*>(traceKernel), "trace", width, height, traceKernelBlockDim, traceKernelGridDim);
}

void Renderer::render()
{
	const Film& film = App::getWindow().getFilm();
	cudaGraphicsResource* filmTextureResource = film.getTextureResource();
	CudaUtils::checkError(cudaGraphicsMapResources(1, &filmTextureResource, 0), "Could not map CUDA texture resource");

	cudaArray_t filmTextureArray;
	CudaUtils::checkError(cudaGraphicsSubResourceGetMappedArray(&filmTextureArray, filmTextureResource, 0, 0), "Could not get CUDA mapped array");

	cudaResourceDesc filmResourceDesc;
	memset(&filmResourceDesc, 0, sizeof(filmResourceDesc));
	filmResourceDesc.resType = cudaResourceTypeArray;
	filmResourceDesc.res.array.array = filmTextureArray;

	cudaSurfaceObject_t filmSurfaceObject;
	CudaUtils::checkError(cudaCreateSurfaceObject(&filmSurfaceObject, &filmResourceDesc), "Could not create CUDA surface object");

	traceKernel<<<traceKernelGridDim, traceKernelBlockDim>>>(camera, nodes, triangles, materials, filmSurfaceObject, film.getWidth(), film.getHeight());
	
	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");

	CudaUtils::checkError(cudaDestroySurfaceObject(filmSurfaceObject), "Could not destroy CUDA surface object");
	CudaUtils::checkError(cudaGraphicsUnmapResources(1, &filmTextureResource, 0), "Could not unmap CUDA texture resource");
}
