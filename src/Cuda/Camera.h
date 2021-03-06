// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Cuda/Math.h"
#include "Cuda/Misc.h"
#include "Core/Camera.h"

using namespace Varjo;

__device__ inline Ray getCameraRay(const CameraData& camera, float2 filmPoint)
{
	float dx = filmPoint.x - camera.halfFilmWidth;
	float dy = filmPoint.y - camera.halfFilmHeight;

	float3 position = camera.filmCenter + (dx * camera.right) + (dy * camera.up);

	Ray ray;
	ray.origin = camera.position;
	ray.direction = normalize(position - camera.position);
	initRay(ray);

	return ray;
}
