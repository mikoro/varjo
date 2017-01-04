// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>

#include "Cuda/Random.h"
#include "Cuda/Math.h"

using namespace Varjo;

__device__ inline uint32_t permute(uint32_t i, uint32_t l, uint32_t p)
{
	uint32_t w = l - 1;

	w |= w >> 1;
	w |= w >> 2;
	w |= w >> 4;
	w |= w >> 8;
	w |= w >> 16;

	do
	{
		i ^= p;
		i *= 0xe170893d;
		i ^= p >> 16;
		i ^= (i & w) >> 4;
		i ^= p >> 8;
		i *= 0x0929eb3f;
		i ^= p >> 23;
		i ^= (i & w) >> 1;
		i *= 1 | p >> 27;
		i *= 0x6935fa69;
		i ^= (i & w) >> 11;
		i *= 0x74dcb303;
		i ^= (i & w) >> 2;
		i *= 0x9e501cc3;
		i ^= (i & w) >> 2;
		i *= 0xc860a3df;
		i &= w;
		i ^= i >> 5;
	} while (i >= l);

	return (i + p) % l;
}

// http://graphics.pixar.com/library/MultiJitteredSampling/
__device__ inline float2 getFilmSample(uint32_t s, uint32_t m, uint32_t n, uint32_t p)
{
	// if s is not permutated, the samples will come out in scanline order
	s = permute(s, m * n, p * 0xa511e9b3);

	uint32_t x = s % m;
	uint32_t y = s / m;

	uint32_t sx = permute(x, m, p * 0xa511e9b3);
	uint32_t sy = permute(y, n, p * 0x63d83595);

	float2 r;

	r.x = float(x) + float(sy) / float(n);
	r.y = float(y) + float(sx) / float(m);

	return r;
}

__device__ inline float3 getCosineSample(Random& random, const float3& normal)
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

__device__ inline float getCosinePdf(const float3& normal, const float3& direction)
{
	return dot(normal, direction) / float(M_PI);
}

__device__ inline float3 getTriangleSample(Random& random, const Triangle& triangle)
{
	float r1 = randomFloat(random);
	float r2 = randomFloat(random);
	float sr1 = sqrtf(r1);

	float u = 1.0f - sr1;
	float v = r2 * sr1;
	float w = 1.0f - u - v;

	return u * triangle.vertices[0] + v * triangle.vertices[1] + w * triangle.vertices[2];
}
