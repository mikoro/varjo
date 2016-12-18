// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Common.h"

/*

pitch = rotation around x-axis [1 0 0]
yaw = rotation around y-axis [0 1 0]
roll = rotation around z-axis [0 0 1]

when all zero, direction points towards negative z-axis [0 0 -1]

angles are in degrees

*/

namespace Varjo
{
	class Vector3;

	class EulerAngle
	{
	public:

		CUDA_CALLABLE explicit EulerAngle(float pitch = 0.0f, float yaw = 0.0f, float roll = 0.0f);

		CUDA_CALLABLE friend EulerAngle operator+(const EulerAngle& e1, const EulerAngle& e2);
		CUDA_CALLABLE friend EulerAngle operator-(const EulerAngle& e1, const EulerAngle& e2);
		CUDA_CALLABLE friend EulerAngle operator*(const EulerAngle& e, float s);
		CUDA_CALLABLE friend EulerAngle operator*(float s, const EulerAngle& e);
		CUDA_CALLABLE friend EulerAngle operator-(const EulerAngle& e);

		CUDA_CALLABLE EulerAngle& operator+=(const EulerAngle& e);
		CUDA_CALLABLE EulerAngle& operator-=(const EulerAngle& e);

		CUDA_CALLABLE void clampPitch();
		CUDA_CALLABLE void normalize();

		CUDA_CALLABLE Vector3 getDirection() const;

		CUDA_CALLABLE static EulerAngle lerp(const EulerAngle& e1, const EulerAngle& e2, float t);

		float pitch; // x-axis
		float yaw; // y-axis
		float roll; // z-axis
	};
}
