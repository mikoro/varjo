// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Ray.h"

using namespace Varjo;

CUDA_CALLABLE void Ray::precalculate()
{
	inverseDirection = direction.inversed();

	directionIsNegative[0] = direction.x < 0.0f;
	directionIsNegative[1] = direction.y < 0.0f;
	directionIsNegative[2] = direction.z < 0.0f;
}
