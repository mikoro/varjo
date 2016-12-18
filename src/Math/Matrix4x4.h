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

		CUDA_CALLABLE Matrix4x4();
		CUDA_CALLABLE Matrix4x4(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23, float m30, float m31, float m32, float m33);
		CUDA_CALLABLE Matrix4x4(const Vector4& r, const Vector4& u, const Vector4& f, const Vector4& t);

		CUDA_CALLABLE friend Matrix4x4 operator+(const Matrix4x4& m, const Matrix4x4& n);
		CUDA_CALLABLE friend Matrix4x4 operator-(const Matrix4x4& m, const Matrix4x4& n);
		CUDA_CALLABLE friend Matrix4x4 operator*(const Matrix4x4& m, float s);
		CUDA_CALLABLE friend Matrix4x4 operator*(float s, const Matrix4x4& m);
		CUDA_CALLABLE friend Matrix4x4 operator*(const Matrix4x4& m, const Matrix4x4& n);
		CUDA_CALLABLE friend Vector4 operator*(const Matrix4x4& m, const Vector4& v);
		CUDA_CALLABLE friend Matrix4x4 operator/(const Matrix4x4& m, float s);
		CUDA_CALLABLE friend Matrix4x4 operator-(const Matrix4x4& m);

		CUDA_CALLABLE friend bool operator==(const Matrix4x4& m, const Matrix4x4& n);
		CUDA_CALLABLE friend bool operator!=(const Matrix4x4& m, const Matrix4x4& n);

		CUDA_CALLABLE Matrix4x4& operator+=(const Matrix4x4& m);
		CUDA_CALLABLE Matrix4x4& operator-=(const Matrix4x4& m);
		CUDA_CALLABLE Matrix4x4& operator*=(const Matrix4x4& m);
		CUDA_CALLABLE Matrix4x4& operator*=(float s);
		CUDA_CALLABLE Matrix4x4& operator/=(float s);

		CUDA_CALLABLE operator float*();
		CUDA_CALLABLE operator const float*() const;

		CUDA_CALLABLE float get(uint32_t row, uint32_t column) const;
		CUDA_CALLABLE void set(uint32_t row, uint32_t column, float value);
		CUDA_CALLABLE Vector4 getRow(uint32_t index) const;
		CUDA_CALLABLE void setRow(uint32_t index, const Vector4& v);
		CUDA_CALLABLE Vector4 getColumn(uint32_t index) const;
		CUDA_CALLABLE void setColumn(uint32_t index, const Vector4& v);

		CUDA_CALLABLE void transpose();
		CUDA_CALLABLE Matrix4x4 transposed() const;
		CUDA_CALLABLE void invert();
		CUDA_CALLABLE Matrix4x4 inverted() const;
		CUDA_CALLABLE bool isZero() const;
		bool isNan() const;

		CUDA_CALLABLE Vector3 transformPosition(const Vector3& v) const;
		CUDA_CALLABLE Vector3 transformDirection(const Vector3& v) const;

		CUDA_CALLABLE static Matrix4x4 scale(const Vector3& s);
		CUDA_CALLABLE static Matrix4x4 scale(float sx, float sy, float sz);
		CUDA_CALLABLE static Matrix4x4 translate(const Vector3& t);
		CUDA_CALLABLE static Matrix4x4 translate(float tx, float ty, float tz);
		CUDA_CALLABLE static Matrix4x4 rotateXYZ(const EulerAngle& e);
		CUDA_CALLABLE static Matrix4x4 rotateZYX(const EulerAngle& e);
		CUDA_CALLABLE static Matrix4x4 rotateXYZ(float pitch, float yaw, float roll);
		CUDA_CALLABLE static Matrix4x4 rotateZYX(float pitch, float yaw, float roll);
		CUDA_CALLABLE static Matrix4x4 rotateX(float angle);
		CUDA_CALLABLE static Matrix4x4 rotateY(float angle);
		CUDA_CALLABLE static Matrix4x4 rotateZ(float angle);
		CUDA_CALLABLE static Matrix4x4 rotate(const Vector3& from, const Vector3& to);

		CUDA_CALLABLE static Matrix4x4 identity();

		float m[4][4];
	};
}
