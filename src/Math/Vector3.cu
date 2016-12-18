// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "tinyformat/tinyformat.h"

#include "Math/Vector3.h"
#include "Math/Vector4.h"
#include "Math/MathUtils.h"

using namespace Varjo;

CUDA_CALLABLE Vector3::Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_)
{
}

CUDA_CALLABLE Vector3::Vector3(const Vector4& v) : x(v.x), y(v.y), z(v.z)
{
}

namespace Varjo
{
	CUDA_CALLABLE Vector3 operator+(const Vector3& v, const Vector3& w)
	{
		return Vector3(v.x + w.x, v.y + w.y, v.z + w.z);
	}

	CUDA_CALLABLE Vector3 operator-(const Vector3& v, const Vector3& w)
	{
		return Vector3(v.x - w.x, v.y - w.y, v.z - w.z);
	}

	CUDA_CALLABLE Vector3 operator*(const Vector3& v, const Vector3& w)
	{
		return Vector3(v.x * w.x, v.y * w.y, v.z * w.z);
	}

	CUDA_CALLABLE Vector3 operator*(const Vector3& v, float s)
	{
		return Vector3(v.x * s, v.y * s, v.z * s);
	}

	CUDA_CALLABLE Vector3 operator*(float s, const Vector3& v)
	{
		return Vector3(v.x * s, v.y * s, v.z * s);
	}

	CUDA_CALLABLE Vector3 operator/(const Vector3& v, const Vector3& w)
	{
		return Vector3(v.x / w.x, v.y / w.y, v.z / w.z);
	}

	CUDA_CALLABLE Vector3 operator/(const Vector3& v, float s)
	{
		float invS = 1.0f / s;
		return Vector3(v.x * invS, v.y * invS, v.z * invS);
	}

	CUDA_CALLABLE Vector3 operator-(const Vector3& v)
	{
		return Vector3(-v.x, -v.y, -v.z);
	}

	CUDA_CALLABLE bool operator==(const Vector3& v, const Vector3& w)
	{
		return MathUtils::almostSame(v.x, w.x) && MathUtils::almostSame(v.y, w.y) && MathUtils::almostSame(v.z, w.z);
	}

	CUDA_CALLABLE bool operator!=(const Vector3& v, const Vector3& w)
	{
		return !(v == w);
	}

	CUDA_CALLABLE bool operator>(const Vector3& v, const Vector3& w)
	{
		return v.x > w.x && v.y > w.y && v.z > w.z;
	}

	CUDA_CALLABLE bool operator<(const Vector3& v, const Vector3& w)
	{
		return v.x < w.x && v.y < w.y && v.z < w.z;
	}
}

CUDA_CALLABLE Vector3& Vector3::operator+=(const Vector3& v)
{
	*this = *this + v;
	return *this;
}

CUDA_CALLABLE Vector3& Vector3::operator-=(const Vector3& v)
{
	*this = *this - v;
	return *this;
}

CUDA_CALLABLE Vector3& Vector3::operator*=(const Vector3& v)
{
	*this = *this * v;
	return *this;
}

CUDA_CALLABLE Vector3& Vector3::operator*=(float s)
{
	*this = *this * s;
	return *this;
}

CUDA_CALLABLE Vector3& Vector3::operator/=(const Vector3& v)
{
	*this = *this / v;
	return *this;
}

CUDA_CALLABLE Vector3& Vector3::operator/=(float s)
{
	*this = *this / s;
	return *this;
}

CUDA_CALLABLE float Vector3::operator[](uint32_t index) const
{
	return (&x)[index];
}

CUDA_CALLABLE float Vector3::getElement(uint32_t index) const
{
	switch (index)
	{
		case 0: return x;
		case 1: return y;
		case 2: return z;
		default: return 0.0f;
	}
}

CUDA_CALLABLE void Vector3::setElement(uint32_t index, float value)
{
	switch (index)
	{
		case 0: x = value;
		case 1: y = value;
		case 2: z = value;
		default: break;
	}
}

CUDA_CALLABLE float Vector3::length() const
{
	return std::sqrt(x * x + y * y + z * z);
}

CUDA_CALLABLE float Vector3::lengthSquared() const
{
	return (x * x + y * y + z * z);
}

CUDA_CALLABLE void Vector3::normalize()
{
	*this /= length();
}

CUDA_CALLABLE Vector3 Vector3::normalized() const
{
	return *this / length();
}

CUDA_CALLABLE void Vector3::inverse()
{
	x = 1.0f / x;
	y = 1.0f / y;
	z = 1.0f / z;
}

CUDA_CALLABLE Vector3 Vector3::inversed() const
{
	Vector3 inverse;

	inverse.x = 1.0f / x;
	inverse.y = 1.0f / y;
	inverse.z = 1.0f / z;

	return inverse;
}

CUDA_CALLABLE bool Vector3::isZero() const
{
	return (x == 0.0f && y == 0.0f && z == 0.0f);
}

bool Vector3::isNan() const
{
	return (std::isnan(x) || std::isnan(y) || std::isnan(z));
}

CUDA_CALLABLE bool Vector3::isNormal() const
{
	return MathUtils::almostSame(lengthSquared(), 1.0f);
}

CUDA_CALLABLE float Vector3::dot(const Vector3& v) const
{
	return (x * v.x) + (y * v.y) + (z * v.z);
}

CUDA_CALLABLE Vector3 Vector3::cross(const Vector3& v) const
{
	Vector3 r;

	r.x = y * v.z - z * v.y;
	r.y = z * v.x - x * v.z;
	r.z = x * v.y - y * v.x;

	return r;
}

CUDA_CALLABLE Vector3 Vector3::reflect(const Vector3& normal) const
{
	return *this - ((2.0f * this->dot(normal)) * normal);
}

CUDA_CALLABLE Vector4 Vector3::toVector4(float w_) const
{
	return Vector4(x, y, z, w_);
}

CUDA_CALLABLE Vector3 Vector3::lerp(const Vector3& v1, const Vector3& v2, float t)
{
	assert(t >= 0.0f && t <= 1.0f);
	return v1 * (1.0f - t) + v2 * t;
}

CUDA_CALLABLE Vector3 Vector3::abs(const Vector3& v)
{
	return Vector3(std::abs(v.x), std::abs(v.y), std::abs(v.z));
}

CUDA_CALLABLE Vector3 Vector3::right()
{
	return Vector3(1.0f, 0.0f, 0.0f);
}

CUDA_CALLABLE Vector3 Vector3::up()
{
	return Vector3(0.0f, 1.0f, 0.0f);
}

CUDA_CALLABLE Vector3 Vector3::forward()
{
	return Vector3(0.0f, 0.0f, 1.0f);
}

CUDA_CALLABLE Vector3 Vector3::almostUp()
{
	return Vector3(0.000001f, 1.0f, 0.000001f);
}

std::string Vector3::toString() const
{
	return tfm::format("(%.2f, %.2f, %.2f)", x, y, z);
}
