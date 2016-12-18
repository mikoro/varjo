// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIfloat, see the LICENSE file.

#pragma once

#include <cstdint>

#include "Common.h"

namespace Varjo
{
	class Vector3;

	class Color
	{
	public:

		CUDA_CALLABLE explicit Color(float r = 0.0f, float g = 0.0f, float b = 0.0f, float a = 1.0f);
		CUDA_CALLABLE explicit Color(int32_t r, int32_t g, int32_t b, int32_t a = 255);

		CUDA_CALLABLE friend Color operator+(const Color& c1, const Color& c2);
		CUDA_CALLABLE friend Color operator-(const Color& c1, const Color& c2);
		CUDA_CALLABLE friend Color operator-(const Color& c);
		CUDA_CALLABLE friend Color operator*(const Color& c1, const Color& c2);
		CUDA_CALLABLE friend Color operator*(const Color& c, float s);
		CUDA_CALLABLE friend Color operator*(float s, const Color& c);
		CUDA_CALLABLE friend Color operator/(const Color& c1, const Color& c2);
		CUDA_CALLABLE friend Color operator/(const Color& c, float s);
		CUDA_CALLABLE friend bool operator==(const Color& c1, const Color& c2);
		CUDA_CALLABLE friend bool operator!=(const Color& c1, const Color& c2);
		
		CUDA_CALLABLE Color& operator+=(const Color& c);
		CUDA_CALLABLE Color& operator-=(const Color& c);
		CUDA_CALLABLE Color& operator*=(const Color& c);
		CUDA_CALLABLE Color& operator*=(float s);
		CUDA_CALLABLE Color& operator/=(float s);

		CUDA_CALLABLE uint32_t getRgbaValue() const;
		CUDA_CALLABLE uint32_t getAbgrValue() const;
		CUDA_CALLABLE float getLuminance() const;
		CUDA_CALLABLE bool isTransparent() const;
		CUDA_CALLABLE bool isZero() const;
		CUDA_CALLABLE bool isClamped() const;
		CUDA_CALLABLE bool isNan() const;
		CUDA_CALLABLE bool isNegative() const;
		CUDA_CALLABLE void clamp();
		CUDA_CALLABLE Color clamped() const;
		CUDA_CALLABLE void clampPositive();
		CUDA_CALLABLE Color clampedPositive() const;
		
		CUDA_CALLABLE static Color fromRgbaValue(uint32_t rgba);
		CUDA_CALLABLE static Color fromAbgrValue(uint32_t abgr);
		CUDA_CALLABLE static Color fromNormal(const Vector3& normal);
		CUDA_CALLABLE static Color fromFloat(float c);
		CUDA_CALLABLE static Color lerp(const Color& start, const Color& end, float alpha);
		CUDA_CALLABLE static Color alphaBlend(const Color& first, const Color& second);
		CUDA_CALLABLE static Color pow(const Color& color, float power);
		CUDA_CALLABLE static Color fastPow(const Color& color, float power);
		CUDA_CALLABLE static Color exp(const Color& color);

		CUDA_CALLABLE static Color red();
		CUDA_CALLABLE static Color green();
		CUDA_CALLABLE static Color blue();
		CUDA_CALLABLE static Color white();
		CUDA_CALLABLE static Color black();

		float r;
		float g;
		float b;
		float a;
	};
}
