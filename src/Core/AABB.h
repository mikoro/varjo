// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Common.h"
#include "Math/Vector3.h"

namespace Varjo
{
	class Ray;
	class EulerAngle;

	class AABB
	{
	public:

		AABB();

		static AABB createFromMinMax(const Vector3& min, const Vector3& max);
		static AABB createFromCenterExtent(const Vector3& center, const Vector3& extent);
		static AABB createFromVertices(const Vector3& v0, const Vector3& v1, const Vector3& v2);

		CUDA_CALLABLE bool intersects(const Ray& ray) const;

		void expand(const AABB& other);

		Vector3 getCenter() const;
		Vector3 getExtent() const;
		float getSurfaceArea() const;

		Vector3 min;
		Vector3 max;
	};
}
