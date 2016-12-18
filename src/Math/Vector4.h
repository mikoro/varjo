// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <string>

#include "Common.h"

namespace Varjo
{
	class Vector3;

	class Vector4
	{
	public:

		CUDA_CALLABLE explicit Vector4(float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 0.0f);
		CUDA_CALLABLE explicit Vector4(const Vector3& v, float w = 0.0f);

		CUDA_CALLABLE friend Vector4 operator+(const Vector4& v, const Vector4& w);
		CUDA_CALLABLE friend Vector4 operator-(const Vector4& v, const Vector4& w);
		CUDA_CALLABLE friend Vector4 operator*(const Vector4& v, const Vector4& w);
		CUDA_CALLABLE friend Vector4 operator*(const Vector4& v, float s);
		CUDA_CALLABLE friend Vector4 operator*(float s, const Vector4& v);
		CUDA_CALLABLE friend Vector4 operator/(const Vector4& v, const Vector4& w);
		CUDA_CALLABLE friend Vector4 operator/(const Vector4& v, float s);
		CUDA_CALLABLE friend Vector4 operator-(const Vector4& v);

		CUDA_CALLABLE friend bool operator==(const Vector4& v, const Vector4& w);
		CUDA_CALLABLE friend bool operator!=(const Vector4& v, const Vector4& w);
		CUDA_CALLABLE friend bool operator>(const Vector4& v, const Vector4& w);
		CUDA_CALLABLE friend bool operator<(const Vector4& v, const Vector4& w);

		CUDA_CALLABLE Vector4& operator+=(const Vector4& v);
		CUDA_CALLABLE Vector4& operator-=(const Vector4& v);
		CUDA_CALLABLE Vector4& operator*=(const Vector4& v);
		CUDA_CALLABLE Vector4& operator*=(float s);
		CUDA_CALLABLE Vector4& operator/=(const Vector4& v);
		CUDA_CALLABLE Vector4& operator/=(float s);
		CUDA_CALLABLE float operator[](uint32_t index) const;

		CUDA_CALLABLE float getElement(uint32_t index) const;
		CUDA_CALLABLE void setElement(uint32_t index, float value);
		CUDA_CALLABLE float length() const;
		CUDA_CALLABLE float lengthSquared() const;
		CUDA_CALLABLE void normalizeLength();
		CUDA_CALLABLE Vector4 normalizedLength() const;
		CUDA_CALLABLE void normalizeForm();
		CUDA_CALLABLE Vector4 normalizedForm() const;
		CUDA_CALLABLE void inverse();
		CUDA_CALLABLE Vector4 inversed() const;
		CUDA_CALLABLE bool isZero() const;
		bool isNan() const;
		CUDA_CALLABLE bool isNormal() const;
		CUDA_CALLABLE float dot(const Vector4& v) const;
		CUDA_CALLABLE Vector3 toVector3() const;

		CUDA_CALLABLE static Vector4 lerp(const Vector4& v1, const Vector4& v2, float t);
		CUDA_CALLABLE static Vector4 abs(const Vector4& v);

		std::string toString() const;

		float x;
		float y;
		float z;
		float w;
	};
}
