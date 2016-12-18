// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cassert>

#include "Math/Matrix4x4.h"
#include "Math/Vector3.h"
#include "Math/Vector4.h"
#include "Math/EulerAngle.h"
#include "Math/MathUtils.h"

using namespace Varjo;

CUDA_CALLABLE Matrix4x4::Matrix4x4()
{
	std::memset(m, 0, sizeof(float) * 16);
}

CUDA_CALLABLE Matrix4x4::Matrix4x4(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23, float m30, float m31, float m32, float m33)
{
	m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
	m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
	m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
	m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
}

CUDA_CALLABLE Matrix4x4::Matrix4x4(const Vector4& r, const Vector4& u, const Vector4& f, const Vector4& t)
{
	m[0][0] = r.x; m[0][1] = u.x; m[0][2] = f.x; m[0][3] = t.x;
	m[1][0] = r.y; m[1][1] = u.y; m[1][2] = f.y; m[1][3] = t.y;
	m[2][0] = r.z; m[2][1] = u.z; m[2][2] = f.z; m[2][3] = t.z;
	m[3][0] = r.w; m[3][1] = u.w; m[3][2] = f.w; m[3][3] = t.w;
}

namespace Varjo
{
	CUDA_CALLABLE Matrix4x4 operator+(const Matrix4x4& m, const Matrix4x4& n)
	{
		Matrix4x4 result;

		for (uint32_t r = 0; r <= 3; ++r)
			for (uint32_t c = 0; c <= 3; ++c)
				result.m[r][c] = m.m[r][c] + n.m[r][c];

		return result;
	}

	CUDA_CALLABLE Matrix4x4 operator-(const Matrix4x4& m, const Matrix4x4& n)
	{
		Matrix4x4 result;

		for (uint32_t r = 0; r <= 3; ++r)
			for (uint32_t c = 0; c <= 3; ++c)
				result.m[r][c] = m.m[r][c] - n.m[r][c];

		return result;
	}

	CUDA_CALLABLE Matrix4x4 operator*(const Matrix4x4& m, float s)
	{
		Matrix4x4 result;

		for (uint32_t r = 0; r <= 3; ++r)
			for (uint32_t c = 0; c <= 3; ++c)
				result.m[r][c] = m.m[r][c] * s;

		return result;
	}

	CUDA_CALLABLE Matrix4x4 operator*(float s, const Matrix4x4& m)
	{
		Matrix4x4 result;

		for (uint32_t r = 0; r <= 3; ++r)
			for (uint32_t c = 0; c <= 3; ++c)
				result.m[r][c] = m.m[r][c] * s;

		return result;
	}

	CUDA_CALLABLE Matrix4x4 operator*(const Matrix4x4& m, const Matrix4x4& n)
	{
		Matrix4x4 result;

		for (uint32_t r = 0; r <= 3; ++r)
			for (uint32_t c = 0; c <= 3; ++c)
				result.m[r][c] = m.m[r][0] * n.m[0][c] + m.m[r][1] * n.m[1][c] + m.m[r][2] * n.m[2][c] + m.m[r][3] * n.m[3][c];

		return result;
	}

	CUDA_CALLABLE Vector4 operator*(const Matrix4x4& m, const Vector4& v)
	{
		Vector4 result;

		result.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3] * v.w;
		result.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3] * v.w;
		result.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3] * v.w;
		result.w = m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3] * v.w;

		return result;
	}

	CUDA_CALLABLE Matrix4x4 operator/(const Matrix4x4& m, float s)
	{
		Matrix4x4 result;

		float invS = 1.0f / s;

		for (uint32_t r = 0; r <= 3; ++r)
			for (uint32_t c = 0; c <= 3; ++c)
				result.m[r][c] = m.m[r][c] * invS;

		return result;
	}

	CUDA_CALLABLE Matrix4x4 operator-(const Matrix4x4& m)
	{
		Matrix4x4 result;

		for (uint32_t r = 0; r <= 3; ++r)
			for (uint32_t c = 0; c <= 3; ++c)
				result.m[r][c] = -m.m[r][c];

		return result;
	}

	CUDA_CALLABLE bool operator==(const Matrix4x4& m, const Matrix4x4& n)
	{
		for (uint32_t r = 0; r <= 3; ++r)
			for (uint32_t c = 0; c <= 3; ++c)
				if (!MathUtils::almostSame(m.m[r][c], n.m[r][c]))
					return false;

		return true;
	}

	CUDA_CALLABLE bool operator!=(const Matrix4x4& m, const Matrix4x4& n)
	{
		return !(m == n);
	}
}

CUDA_CALLABLE Matrix4x4& Matrix4x4::operator+=(const Matrix4x4& n)
{
	*this = *this + n;
	return *this;
}

CUDA_CALLABLE Matrix4x4& Matrix4x4::operator-=(const Matrix4x4& n)
{
	*this = *this - n;
	return *this;
}

CUDA_CALLABLE Matrix4x4& Matrix4x4::operator*=(const Matrix4x4& n)
{
	*this = *this * n;
	return *this;
}

CUDA_CALLABLE Matrix4x4& Matrix4x4::operator*=(float s)
{
	*this = *this * s;
	return *this;
}

CUDA_CALLABLE Matrix4x4& Matrix4x4::operator/=(float s)
{
	*this = *this / s;
	return *this;
}

CUDA_CALLABLE Matrix4x4::operator float*()
{
	return &m[0][0];
}

CUDA_CALLABLE Matrix4x4::operator const float*() const
{
	return &m[0][0];
}

CUDA_CALLABLE float Matrix4x4::get(uint32_t row, uint32_t column) const
{
	assert(row <= 3 && column <= 3);
	return m[row][column];
}

CUDA_CALLABLE void Matrix4x4::set(uint32_t row, uint32_t column, float value)
{
	assert(row <= 3 && column <= 3);
	m[row][column] = value;
}

CUDA_CALLABLE Vector4 Matrix4x4::getRow(uint32_t index) const
{
	assert(index <= 3);
	return Vector4(m[index][0], m[index][1], m[index][2], m[index][3]);
}

CUDA_CALLABLE void Matrix4x4::setRow(uint32_t index, const Vector4& v)
{
	assert(index <= 3);

	m[index][0] = v.x;
	m[index][1] = v.y;
	m[index][2] = v.z;
	m[index][3] = v.w;
}

CUDA_CALLABLE Vector4 Matrix4x4::getColumn(uint32_t index) const
{
	assert(index <= 3);
	return Vector4(m[0][index], m[1][index], m[2][index], m[3][index]);
}

CUDA_CALLABLE void Matrix4x4::setColumn(uint32_t index, const Vector4& v)
{
	assert(index <= 3);

	m[0][index] = v.x;
	m[1][index] = v.y;
	m[2][index] = v.z;
	m[3][index] = v.w;
}

CUDA_CALLABLE void Matrix4x4::transpose()
{
	*this = transposed();
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::transposed() const
{
	Matrix4x4 result;

	for (uint32_t r = 0; r <= 3; ++r)
		for (uint32_t c = 0; c <= 3; ++c)
			result.m[r][c] = m[c][r];

	return result;
}

CUDA_CALLABLE void Matrix4x4::invert()
{
	*this = inverted();
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::inverted() const
{
	float n[16], inv[16], out[16];
	std::memcpy(n, m, sizeof(float) * 16);

	inv[0] = n[5] * n[10] * n[15] - n[5] * n[11] * n[14] - n[9] * n[6] * n[15] + n[9] * n[7] * n[14] + n[13] * n[6] * n[11] - n[13] * n[7] * n[10];
	inv[4] = -n[4] * n[10] * n[15] + n[4] * n[11] * n[14] + n[8] * n[6] * n[15] - n[8] * n[7] * n[14] - n[12] * n[6] * n[11] + n[12] * n[7] * n[10];
	inv[8] = n[4] * n[9] * n[15] - n[4] * n[11] * n[13] - n[8] * n[5] * n[15] + n[8] * n[7] * n[13] + n[12] * n[5] * n[11] - n[12] * n[7] * n[9];
	inv[12] = -n[4] * n[9] * n[14] + n[4] * n[10] * n[13] + n[8] * n[5] * n[14] - n[8] * n[6] * n[13] - n[12] * n[5] * n[10] + n[12] * n[6] * n[9];
	inv[1] = -n[1] * n[10] * n[15] + n[1] * n[11] * n[14] + n[9] * n[2] * n[15] - n[9] * n[3] * n[14] - n[13] * n[2] * n[11] + n[13] * n[3] * n[10];
	inv[5] = n[0] * n[10] * n[15] - n[0] * n[11] * n[14] - n[8] * n[2] * n[15] + n[8] * n[3] * n[14] + n[12] * n[2] * n[11] - n[12] * n[3] * n[10];
	inv[9] = -n[0] * n[9] * n[15] + n[0] * n[11] * n[13] + n[8] * n[1] * n[15] - n[8] * n[3] * n[13] - n[12] * n[1] * n[11] + n[12] * n[3] * n[9];
	inv[13] = n[0] * n[9] * n[14] - n[0] * n[10] * n[13] - n[8] * n[1] * n[14] + n[8] * n[2] * n[13] + n[12] * n[1] * n[10] - n[12] * n[2] * n[9];
	inv[2] = n[1] * n[6] * n[15] - n[1] * n[7] * n[14] - n[5] * n[2] * n[15] + n[5] * n[3] * n[14] + n[13] * n[2] * n[7] - n[13] * n[3] * n[6];
	inv[6] = -n[0] * n[6] * n[15] + n[0] * n[7] * n[14] + n[4] * n[2] * n[15] - n[4] * n[3] * n[14] - n[12] * n[2] * n[7] + n[12] * n[3] * n[6];
	inv[10] = n[0] * n[5] * n[15] - n[0] * n[7] * n[13] - n[4] * n[1] * n[15] + n[4] * n[3] * n[13] + n[12] * n[1] * n[7] - n[12] * n[3] * n[5];
	inv[14] = -n[0] * n[5] * n[14] + n[0] * n[6] * n[13] + n[4] * n[1] * n[14] - n[4] * n[2] * n[13] - n[12] * n[1] * n[6] + n[12] * n[2] * n[5];
	inv[3] = -n[1] * n[6] * n[11] + n[1] * n[7] * n[10] + n[5] * n[2] * n[11] - n[5] * n[3] * n[10] - n[9] * n[2] * n[7] + n[9] * n[3] * n[6];
	inv[7] = n[0] * n[6] * n[11] - n[0] * n[7] * n[10] - n[4] * n[2] * n[11] + n[4] * n[3] * n[10] + n[8] * n[2] * n[7] - n[8] * n[3] * n[6];
	inv[11] = -n[0] * n[5] * n[11] + n[0] * n[7] * n[9] + n[4] * n[1] * n[11] - n[4] * n[3] * n[9] - n[8] * n[1] * n[7] + n[8] * n[3] * n[5];
	inv[15] = n[0] * n[5] * n[10] - n[0] * n[6] * n[9] - n[4] * n[1] * n[10] + n[4] * n[2] * n[9] + n[8] * n[1] * n[6] - n[8] * n[2] * n[5];

	Matrix4x4 result;

	float det = n[0] * inv[0] + n[1] * inv[4] + n[2] * inv[8] + n[3] * inv[12];

	if (det == 0.0f)
		return result;

	det = 1.0f / det;

	for (uint32_t i = 0; i < 16; i++)
		out[i] = inv[i] * det;

	std::memcpy(result.m, out, sizeof(float) * 16);
	return result;
}

CUDA_CALLABLE bool Matrix4x4::isZero() const
{
	for (uint32_t r = 0; r <= 3; ++r)
	{
		for (uint32_t c = 0; c <= 3; ++c)
		{
			if (m[r][c] != 0.0f)
				return false;
		}
	}

	return true;
}

bool Matrix4x4::isNan() const
{
	for (uint32_t r = 0; r <= 3; ++r)
	{
		for (uint32_t c = 0; c <= 3; ++c)
		{
			if (std::isnan(m[r][c]))
				return true;
		}
	}

	return false;
}

CUDA_CALLABLE Vector3 Matrix4x4::transformPosition(const Vector3& v) const
{
	Vector3 result;

	result.x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3];
	result.y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3];
	result.z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3];

	return result;
}

CUDA_CALLABLE Vector3 Matrix4x4::transformDirection(const Vector3& v) const
{
	Vector3 result;

	result.x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z;
	result.y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z;
	result.z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z;

	return result;
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::scale(const Vector3& s)
{
	return scale(s.x, s.y, s.z);
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::scale(float sx, float sy, float sz)
{
	Matrix4x4 result = identity();

	result.m[0][0] = sx;
	result.m[1][1] = sy;
	result.m[2][2] = sz;

	return result;
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::translate(const Vector3& t)
{
	return translate(t.x, t.y, t.z);
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::translate(float tx, float ty, float tz)
{
	Matrix4x4 result = identity();

	result.m[0][3] = tx;
	result.m[1][3] = ty;
	result.m[2][3] = tz;

	return result;
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::rotateXYZ(const EulerAngle& e)
{
	return rotateXYZ(e.pitch, e.yaw, e.roll);
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::rotateZYX(const EulerAngle& e)
{
	return rotateZYX(e.pitch, e.yaw, e.roll);
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::rotateXYZ(float pitch, float yaw, float roll)
{
	return rotateX(pitch) * rotateY(yaw) * rotateZ(roll);
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::rotateZYX(float pitch, float yaw, float roll)
{
	return rotateZ(roll) * rotateY(yaw) * rotateX(pitch);
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::rotateX(float angle)
{
	Matrix4x4 result = identity();

	float sine = sin(MathUtils::degToRad(angle));
	float cosine = cos(MathUtils::degToRad(angle));

	result.m[1][1] = cosine;
	result.m[1][2] = -sine;
	result.m[2][1] = sine;
	result.m[2][2] = cosine;

	return result;
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::rotateY(float angle)
{
	Matrix4x4 result = identity();

	float sine = sin(MathUtils::degToRad(angle));
	float cosine = cos(MathUtils::degToRad(angle));

	result.m[0][0] = cosine;
	result.m[0][2] = sine;
	result.m[2][0] = -sine;
	result.m[2][2] = cosine;

	return result;
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::rotateZ(float angle)
{
	Matrix4x4 result = identity();

	float sine = sin(MathUtils::degToRad(angle));
	float cosine = cos(MathUtils::degToRad(angle));

	result.m[0][0] = cosine;
	result.m[0][1] = -sine;
	result.m[1][0] = sine;
	result.m[1][1] = cosine;

	return result;
}

// http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
CUDA_CALLABLE Matrix4x4 Matrix4x4::rotate(const Vector3& from, const Vector3& to)
{
	Vector3 v = from.cross(to);
	float s = v.length();
	float c = from.dot(to);

	Matrix4x4 vx(
		0.0f, -v.z, v.y, 0.0f,
		v.z, 0.0f, -v.x, 0.0f,
		-v.y, v.x, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f);

	if (s != 0.0f)
		return identity() + vx + (vx * vx) * ((1.0f - c) / (s * s));
	else
		return identity();
}

CUDA_CALLABLE Matrix4x4 Matrix4x4::identity()
{
	return Matrix4x4(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
}
