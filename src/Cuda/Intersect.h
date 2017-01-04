// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "Cuda/Math.h"
#include "Core/Triangle.h"
#include "Core/BVH.h"

using namespace Varjo;

// http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
__device__ inline void intersectTriangle(const Triangle* __restrict triangles, uint32_t triangleIndex, const Ray& ray, Intersection& intersection)
{
	const Triangle& triangle = triangles[triangleIndex];

	float3 v0v1 = triangle.vertices[1] - triangle.vertices[0];
	float3 v0v2 = triangle.vertices[2] - triangle.vertices[0];

	float3 pvec = cross(ray.direction, v0v2);
	float determinant = dot(v0v1, pvec);

	if (determinant == 0.0f)
		return;

	float invDeterminant = 1.0f / determinant;

	float3 tvec = ray.origin - triangle.vertices[0];
	float u = dot(tvec, pvec) * invDeterminant;

	if (u < 0.0f || u > 1.0f)
		return;

	float3 qvec = cross(tvec, v0v1);
	float v = dot(ray.direction, qvec) * invDeterminant;

	if (v < 0.0f || (u + v) > 1.0f)
		return;

	float distance = dot(v0v2, qvec) * invDeterminant;

	if (distance < 0.0f || distance < ray.minDistance || distance > ray.maxDistance || distance > intersection.distance)
		return;

	float w = 1.0f - u - v;

	intersection.wasFound = true;
	intersection.distance = distance;
	intersection.position = ray.origin + (distance * ray.direction);
	intersection.normal = w * triangle.normals[0] + u * triangle.normals[1] + v * triangle.normals[2];
	intersection.texcoord = w * triangle.texcoords[0] + u * triangle.texcoords[1] + v * triangle.texcoords[2];
	intersection.triangleIndex = triangleIndex;
	intersection.materialIndex = triangle.materialIndex;
}

// https://mediatech.aalto.fi/~timo/publications/aila2012hpg_techrep.pdf
__device__ inline bool intersectAabb(const AABB& aabb, const Ray& ray)
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

__device__ inline void intersectBvh(const BVHNode* __restrict nodes, const Triangle* __restrict triangles, const Ray& ray, Intersection& intersection)
{
	BVHNode node;
	uint32_t stack[16];
	uint32_t stackIndex = 1;
	stack[0] = 0;

	while (stackIndex > 0)
	{
		uint32_t nodeIndex = stack[--stackIndex];
		node = nodes[nodeIndex];

		// leaf node
		if (node.rightOffset == 0)
		{
			for (int i = 0; i < node.triangleCount; ++i)
				intersectTriangle(triangles, node.triangleOffset + i, ray, intersection);

			continue;
		}

		if (intersectAabb(node.aabb, ray))
		{
			stack[stackIndex++] = nodeIndex + 1; // left child
			stack[stackIndex++] = nodeIndex + uint32_t(node.rightOffset); // right child
		}
	}
}
