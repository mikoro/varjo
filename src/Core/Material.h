// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cuda_runtime.h>

namespace Varjo
{
	struct Material
	{
		float3 baseColor;
		float subsurface = 0.0f;
		float metallic = 0.0f;
		float specular = 0.0f;
		float specularTint = 0.0f;
		float roughness = 0.0f;
		float anisotropic = 0.0f;
		float sheen = 0.0f;
		float sheenTint = 0.0f;
		float clearcoat = 0.0f;
		float clearcoatGloss = 0.0f;
	};
}
