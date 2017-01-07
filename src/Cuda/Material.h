// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Cuda/Random.h"
#include "Cuda/Math.h"

using namespace Varjo;

__device__ inline float3 getDiffuseBrdf(const Material& material)
{
	return material.baseColor / float(M_PI);
}

__device__ inline float3 getDiffuseDirection(Random& random, const float3& normal)
{
	float u1 = randomFloat(random);
	float u2 = randomFloat(random);
	float r = sqrtf(u1);
	float theta = 2.0f * float(M_PI) * u2;
	float x = r * cosf(theta);
	float y = r * sinf(theta);
	float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

	float3 u = normalize(cross(normal, make_float3(0.0001f, 1.0000f, 0.0001f)));
	float3 v = normalize(cross(u, normal));

	return x * u + y * v + z * normal;
}

__device__ inline float getDiffusePdf(const float3& normal, const float3& direction)
{
	return dot(normal, direction) / float(M_PI);
}
