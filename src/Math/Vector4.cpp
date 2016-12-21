// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "tinyformat/tinyformat.h"

#include "Math/Vector4.h"
#include "Math/Vector3.h"
#include "Math/MathUtils.h"

using namespace Varjo;

Vector4::Vector4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_)
{
}

Vector4::Vector4(const Vector3& v, float w_) : x(v.x), y(v.y), z(v.z), w(w_)
{
}

namespace Varjo
{
	Vector4 operator+(const Vector4& v, const Vector4& w)
	{
		return Vector4(v.x + w.x, v.y + w.y, v.z + w.z, v.w + w.w);
	}

	Vector4 operator-(const Vector4& v, const Vector4& w)
	{
		return Vector4(v.x - w.x, v.y - w.y, v.z - w.z, v.w - w.w);
	}

	Vector4 operator*(const Vector4& v, const Vector4& w)
	{
		return Vector4(v.x * w.x, v.y * w.y, v.z * w.z, v.w * w.w);
	}

	Vector4 operator*(const Vector4& v, float s)
	{
		return Vector4(v.x * s, v.y * s, v.z * s, v.w * s);
	}

	Vector4 operator*(float s, const Vector4& v)
	{
		return Vector4(v.x * s, v.y * s, v.z * s, v.w * s);
	}

	Vector4 operator/(const Vector4& v, const Vector4& w)
	{
		return Vector4(v.x / w.x, v.y / w.y, v.z / w.z, v.w / w.w);
	}

	Vector4 operator/(const Vector4& v, float s)
	{
		float invS = 1.0f / s;
		return Vector4(v.x * invS, v.y * invS, v.z * invS, v.w * invS);
	}

	Vector4 operator-(const Vector4& v)
	{
		return Vector4(-v.x, -v.y, -v.z, -v.w);
	}

	bool operator==(const Vector4& v, const Vector4& w)
	{
		return MathUtils::almostSame(v.x, w.x) && MathUtils::almostSame(v.y, w.y) && MathUtils::almostSame(v.z, w.z) && MathUtils::almostSame(v.w, w.w);
	}

	bool operator!=(const Vector4& v, const Vector4& w)
	{
		return !(v == w);
	}

	bool operator>(const Vector4& v, const Vector4& w)
	{
		return v.x > w.x && v.y > w.y && v.z > w.z && v.w > w.w;
	}

	bool operator<(const Vector4& v, const Vector4& w)
	{
		return v.x < w.x && v.y < w.y && v.z < w.z && v.w < w.w;
	}
}

Vector4& Vector4::operator+=(const Vector4& v)
{
	*this = *this + v;
	return *this;
}

Vector4& Vector4::operator-=(const Vector4& v)
{
	*this = *this - v;
	return *this;
}

Vector4& Vector4::operator*=(const Vector4& v)
{
	*this = *this * v;
	return *this;
}

Vector4& Vector4::operator*=(float s)
{
	*this = *this * s;
	return *this;
}

Vector4& Vector4::operator/=(const Vector4& v)
{
	*this = *this / v;
	return *this;
}

Vector4& Vector4::operator/=(float s)
{
	*this = *this / s;
	return *this;
}

float Vector4::operator[](uint32_t index) const
{
	return (&x)[index];
}

float Vector4::getElement(uint32_t index) const
{
	switch (index)
	{
		case 0: return x;
		case 1: return y;
		case 2: return z;
		case 3: return w;
		default: return 0.0f;
	}
}

void Vector4::setElement(uint32_t index, float value)
{
	switch (index)
	{
		case 0: x = value;
		case 1: y = value;
		case 2: z = value;
		case 3: w = value;
		default: break;
	}
}

float Vector4::length() const
{
	return std::sqrt(x * x + y * y + z * z + w * w);
}

float Vector4::lengthSquared() const
{
	return (x * x + y * y + z * z + w * w);
}

void Vector4::normalizeLength()
{
	*this /= length();
}

Vector4 Vector4::normalizedLength() const
{
	return *this / length();
}

void Vector4::normalizeForm()
{
	*this /= w;
}

Vector4 Vector4::normalizedForm() const
{
	return *this / w;
}

void Vector4::inverse()
{
	x = 1.0f / x;
	y = 1.0f / y;
	z = 1.0f / z;
	w = 1.0f / w;
}

Vector4 Vector4::inversed() const
{
	Vector4 inverse;

	inverse.x = 1.0f / x;
	inverse.y = 1.0f / y;
	inverse.z = 1.0f / z;
	inverse.w = 1.0f / w;

	return inverse;
}

bool Vector4::isZero() const
{
	return (x == 0.0f && y == 0.0f && z == 0.0f && w == 0.0f);
}

bool Vector4::isNan() const
{
	return (std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(w));
}

bool Vector4::isNormal() const
{
	return MathUtils::almostSame(lengthSquared(), 1.0f);
}

float Vector4::dot(const Vector4& v) const
{
	return (x * v.x) + (y * v.y) + (z * v.z) + (w * v.w);
}

Vector3 Vector4::toVector3() const
{
	return Vector3(x, y, z);
}

Vector4 Vector4::lerp(const Vector4& v1, const Vector4& v2, float t)
{
	assert(t >= 0.0f && t <= 1.0f);
	return v1 * (1.0f - t) + v2 * t;
}

Vector4 Vector4::abs(const Vector4& v)
{
	return Vector4(std::abs(v.x), std::abs(v.y), std::abs(v.z), std::abs(v.w));
}

std::string Vector4::toString() const
{
	return tfm::format("(%.2f, %.2f, %.2f, %.2f)", x, y, z, w);
}
