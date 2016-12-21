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

		explicit Vector3(float x = 0.0f, float y = 0.0f, float z = 0.0f);
		explicit Vector3(const Vector4& v);

		friend Vector3 operator+(const Vector3& v, const Vector3& w);
		friend Vector3 operator-(const Vector3& v, const Vector3& w);
		friend Vector3 operator*(const Vector3& v, const Vector3& w);
		friend Vector3 operator*(const Vector3& v, float s);
		friend Vector3 operator*(float s, const Vector3& v);
		friend Vector3 operator/(const Vector3& v, const Vector3& w);
		friend Vector3 operator/(const Vector3& v, float s);
		friend Vector3 operator-(const Vector3& v);

		friend bool operator==(const Vector3& v, const Vector3& w);
		friend bool operator!=(const Vector3& v, const Vector3& w);
		friend bool operator>(const Vector3& v, const Vector3& w);
		friend bool operator<(const Vector3& v, const Vector3& w);

		Vector3& operator+=(const Vector3& v);
		Vector3& operator-=(const Vector3& v);
		Vector3& operator*=(const Vector3& v);
		Vector3& operator*=(float s);
		Vector3& operator/=(const Vector3& v);
		Vector3& operator/=(float s);
		float operator[](uint32_t index) const;

		float getElement(uint32_t index) const;
		void setElement(uint32_t index, float value);
		float length() const;
		float lengthSquared() const;
		void normalize();
		Vector3 normalized() const;
		void inverse();
		Vector3 inversed() const;
		bool isZero() const;
		bool isNan() const;
		bool isNormal() const;
		float dot(const Vector3& v) const;
		Vector3 cross(const Vector3& v) const;
		Vector3 reflect(const Vector3& normal) const;
		Vector4 toVector4(float w = 0.0f) const;

		static Vector3 lerp(const Vector3& v1, const Vector3& v2, float t);
		static Vector3 abs(const Vector3& v);

		static Vector3 right();
		static Vector3 up();
		static Vector3 forward();
		static Vector3 almostUp();

		std::string toString() const;

		float x;
		float y;
		float z;
	};
}
