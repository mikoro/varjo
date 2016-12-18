// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Common.h"
#include "Math/Vector3.h"

/*

OrthoNormal Basis

left-handed coordinate system
-z is towards the monitor

u = right
v = up
w = forward (normal)

*/

namespace Varjo
{
	class Matrix4x4;

	class ONB
	{
	public:

		CUDA_CALLABLE ONB();
		CUDA_CALLABLE ONB(const Vector3& u, const Vector3& v, const Vector3& w);

		CUDA_CALLABLE ONB transformed(const Matrix4x4& tranformation) const;

		CUDA_CALLABLE static ONB fromNormal(const Vector3& normal, const Vector3& up = Vector3::almostUp());

		CUDA_CALLABLE static ONB up();

		Vector3 u;
		Vector3 v;
		Vector3 w;
	};
}
