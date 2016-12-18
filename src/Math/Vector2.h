// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <string>

#include "Common.h"

namespace Varjo
{
	class Vector2
	{
	public:

		CUDA_CALLABLE explicit Vector2(float x = 0.0f, float y = 0.0f);

		CUDA_CALLABLE friend Vector2 operator+(const Vector2& v, const Vector2& w);
		CUDA_CALLABLE friend Vector2 operator-(const Vector2& v, const Vector2& w);
		CUDA_CALLABLE friend Vector2 operator*(const Vector2& v, const Vector2& w);
		CUDA_CALLABLE friend Vector2 operator*(const Vector2& v, float s);
		CUDA_CALLABLE friend Vector2 operator*(float s, const Vector2& v);
		CUDA_CALLABLE friend Vector2 operator/(const Vector2& v, const Vector2& w);
		CUDA_CALLABLE friend Vector2 operator/(const Vector2& v, float s);
		CUDA_CALLABLE friend Vector2 operator-(const Vector2& v);

		CUDA_CALLABLE friend bool operator==(const Vector2& v, const Vector2& w);
		CUDA_CALLABLE friend bool operator!=(const Vector2& v, const Vector2& w);
		CUDA_CALLABLE friend bool operator>(const Vector2& v, const Vector2& w);
		CUDA_CALLABLE friend bool operator<(const Vector2& v, const Vector2& w);

		CUDA_CALLABLE Vector2& operator+=(const Vector2& v);
		CUDA_CALLABLE Vector2& operator-=(const Vector2& v);
		CUDA_CALLABLE Vector2& operator*=(const Vector2& v);
		CUDA_CALLABLE Vector2& operator*=(float s);
		CUDA_CALLABLE Vector2& operator/=(const Vector2& v);
		CUDA_CALLABLE Vector2& operator/=(float s);
		CUDA_CALLABLE float operator[](uint32_t index) const;

		CUDA_CALLABLE float getElement(uint32_t index) const;
		CUDA_CALLABLE void setElement(uint32_t index, float value);
		CUDA_CALLABLE float length() const;
		CUDA_CALLABLE float lengthSquared() const;
		CUDA_CALLABLE void normalize();
		CUDA_CALLABLE Vector2 normalized() const;
		CUDA_CALLABLE void inverse();
		CUDA_CALLABLE Vector2 inversed() const;
		CUDA_CALLABLE bool isZero() const;
		bool isNan() const;
		CUDA_CALLABLE bool isNormal() const;
		CUDA_CALLABLE float dot(const Vector2& v) const;
		CUDA_CALLABLE Vector2 reflect(const Vector2& normal) const;

		CUDA_CALLABLE static Vector2 lerp(const Vector2& v1, const Vector2& v2, float t);
		CUDA_CALLABLE static Vector2 abs(const Vector2& v);

		std::string toString() const;

		float x;
		float y;
	};
}
