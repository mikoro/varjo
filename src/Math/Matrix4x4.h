// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>

#include "Common.h"

/*

https://en.wikipedia.org/wiki/Row-major_order

row major memory data order
basis vectors are stored as columns
needs to be transposed when sent to opengl

array indices:
m[row][column]
[00, 01, 02, 03]
[10, 11, 12, 13]
[20, 21, 22, 23]
[30, 31, 32, 33]
 R   U   F   T

R = [1 0 0] (x, right)
U = [0 1 0] (y, up)
F = [0 0 1] (z, forward)
T = translation

*/

namespace Varjo
{
	class Vector3;
	class Vector4;
	class EulerAngle;

	class Matrix4x4
	{
	public:

		Matrix4x4();
		Matrix4x4(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23, float m30, float m31, float m32, float m33);
		Matrix4x4(const Vector4& r, const Vector4& u, const Vector4& f, const Vector4& t);

		friend Matrix4x4 operator+(const Matrix4x4& m, const Matrix4x4& n);
		friend Matrix4x4 operator-(const Matrix4x4& m, const Matrix4x4& n);
		friend Matrix4x4 operator*(const Matrix4x4& m, float s);
		friend Matrix4x4 operator*(float s, const Matrix4x4& m);
		friend Matrix4x4 operator*(const Matrix4x4& m, const Matrix4x4& n);
		friend Vector4 operator*(const Matrix4x4& m, const Vector4& v);
		friend Matrix4x4 operator/(const Matrix4x4& m, float s);
		friend Matrix4x4 operator-(const Matrix4x4& m);

		friend bool operator==(const Matrix4x4& m, const Matrix4x4& n);
		friend bool operator!=(const Matrix4x4& m, const Matrix4x4& n);

		Matrix4x4& operator+=(const Matrix4x4& m);
		Matrix4x4& operator-=(const Matrix4x4& m);
		Matrix4x4& operator*=(const Matrix4x4& m);
		Matrix4x4& operator*=(float s);
		Matrix4x4& operator/=(float s);

		operator float*();
		operator const float*() const;

		float get(uint32_t row, uint32_t column) const;
		void set(uint32_t row, uint32_t column, float value);
		Vector4 getRow(uint32_t index) const;
		void setRow(uint32_t index, const Vector4& v);
		Vector4 getColumn(uint32_t index) const;
		void setColumn(uint32_t index, const Vector4& v);

		void transpose();
		Matrix4x4 transposed() const;
		void invert();
		Matrix4x4 inverted() const;
		bool isZero() const;
		bool isNan() const;

		Vector3 transformPosition(const Vector3& v) const;
		Vector3 transformDirection(const Vector3& v) const;

		static Matrix4x4 scale(const Vector3& s);
		static Matrix4x4 scale(float sx, float sy, float sz);
		static Matrix4x4 translate(const Vector3& t);
		static Matrix4x4 translate(float tx, float ty, float tz);
		static Matrix4x4 rotateXYZ(const EulerAngle& e);
		static Matrix4x4 rotateZYX(const EulerAngle& e);
		static Matrix4x4 rotateXYZ(float pitch, float yaw, float roll);
		static Matrix4x4 rotateZYX(float pitch, float yaw, float roll);
		static Matrix4x4 rotateX(float angle);
		static Matrix4x4 rotateY(float angle);
		static Matrix4x4 rotateZ(float angle);
		static Matrix4x4 rotate(const Vector3& from, const Vector3& to);

		static Matrix4x4 identity();

		float m[4][4];
	};
}
