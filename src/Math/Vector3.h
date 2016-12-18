// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <string>

#include "Common.h"

namespace Varjo
{
	class Vector4;

	class Vector3
	{
	public:

		CUDA_CALLABLE explicit Vector3(float x = 0.0f, float y = 0.0f, float z = 0.0f);
		CUDA_CALLABLE explicit Vector3(const Vector4& v);

		CUDA_CALLABLE friend Vector3 operator+(const Vector3& v, const Vector3& w);
		CUDA_CALLABLE friend Vector3 operator-(const Vector3& v, const Vector3& w);
		CUDA_CALLABLE friend Vector3 operator*(const Vector3& v, const Vector3& w);
		CUDA_CALLABLE friend Vector3 operator*(const Vector3& v, float s);
		CUDA_CALLABLE friend Vector3 operator*(float s, const Vector3& v);
		CUDA_CALLABLE friend Vector3 operator/(const Vector3& v, const Vector3& w);
		CUDA_CALLABLE friend Vector3 operator/(const Vector3& v, float s);
		CUDA_CALLABLE friend Vector3 operator-(const Vector3& v);

		CUDA_CALLABLE friend bool operator==(const Vector3& v, const Vector3& w);
		CUDA_CALLABLE friend bool operator!=(const Vector3& v, const Vector3& w);
		CUDA_CALLABLE friend bool operator>(const Vector3& v, const Vector3& w);
		CUDA_CALLABLE friend bool operator<(const Vector3& v, const Vector3& w);

		CUDA_CALLABLE Vector3& operator+=(const Vector3& v);
		CUDA_CALLABLE Vector3& operator-=(const Vector3& v);
		CUDA_CALLABLE Vector3& operator*=(const Vector3& v);
		CUDA_CALLABLE Vector3& operator*=(float s);
		CUDA_CALLABLE Vector3& operator/=(const Vector3& v);
		CUDA_CALLABLE Vector3& operator/=(float s);
		CUDA_CALLABLE float operator[](uint32_t index) const;

		CUDA_CALLABLE float getElement(uint32_t index) const;
		CUDA_CALLABLE void setElement(uint32_t index, float value);
		CUDA_CALLABLE float length() const;
		CUDA_CALLABLE float lengthSquared() const;
		CUDA_CALLABLE void normalize();
		CUDA_CALLABLE Vector3 normalized() const;
		CUDA_CALLABLE void inverse();
		CUDA_CALLABLE Vector3 inversed() const;
		CUDA_CALLABLE bool isZero() const;
		bool isNan() const;
		CUDA_CALLABLE bool isNormal() const;
		CUDA_CALLABLE float dot(const Vector3& v) const;
		CUDA_CALLABLE Vector3 cross(const Vector3& v) const;
		CUDA_CALLABLE Vector3 reflect(const Vector3& normal) const;
		CUDA_CALLABLE Vector4 toVector4(float w = 0.0f) const;

		CUDA_CALLABLE static Vector3 lerp(const Vector3& v1, const Vector3& v2, float t);
		CUDA_CALLABLE static Vector3 abs(const Vector3& v);

		CUDA_CALLABLE static Vector3 right();
		CUDA_CALLABLE static Vector3 up();
		CUDA_CALLABLE static Vector3 forward();
		CUDA_CALLABLE static Vector3 almostUp();

		std::string toString() const;

		float x;
		float y;
		float z;
	};
}
