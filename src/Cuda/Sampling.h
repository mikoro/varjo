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

__device__ inline void generateLightSample(
	Paths* __restrict paths,
	const Triangle* __restrict triangles,
	const uint32_t* __restrict emitters,
	const Material* __restrict materials,
	Random& random,
	uint32_t pathId,
	uint32_t emitterCount)
{
	// generate random barycentric coordinates
	float r1 = randomFloat(random);
	float r2 = randomFloat(random);
	float sr1 = sqrtf(r1);
	float u = 1.0f - sr1;
	float v = r2 * sr1;
	float w = 1.0f - u - v;

	// select one random emissive triangle
	uint32_t triangleIndex = emitters[randomInt(random, emitterCount)];
	const Triangle& triangle = triangles[triangleIndex];
	const Material& material = materials[triangle.materialIndex];

	float3 lightPosition = u * triangle.vertices[0] + v * triangle.vertices[1] + w * triangle.vertices[2];
	float3 lightNormal = u * triangle.normals[0] + v * triangle.normals[1] + w * triangle.normals[2];
	//float2 lightTexcoord = u * triangle.texcoords[0] + v * triangle.texcoords[1] + w * triangle.texcoords[2];
	float3 extensionIntersectionNormal = paths->extensionIntersection[pathId].normal;
	
	float3 startPosition = paths->extensionIntersection[pathId].position + RAY_EPSILON * extensionIntersectionNormal;
	float3 toLight = lightPosition - startPosition;
	float3 toLightDir = normalize(toLight);
	float toLightDist = length(toLight);

	Ray lightRay;
	lightRay.origin = startPosition;
	lightRay.direction = toLightDir;
	lightRay.maxDistance = toLightDist - RAY_EPSILON;
	initRay(lightRay);

	paths->lightRay[pathId] = lightRay;
	paths->lightEmittance[pathId] = material.emittance;

	float tempDot = dot(lightNormal, -lightRay.direction);

	if (tempDot > 0.0f)
		paths->lightPdf[pathId] = (toLightDist * toLightDist) / tempDot;
	else
		paths->lightPdf[pathId] = 0.0f;

	tempDot = dot(extensionIntersectionNormal, lightRay.direction);

	if (tempDot > 0.0f)
		paths->lightCosine[pathId] = tempDot;
	else
		paths->lightCosine[pathId] = 0.0f;
}
