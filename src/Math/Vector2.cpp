// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "tinyformat/tinyformat.h"

#include "Math/Vector2.h"
#include "Math/MathUtils.h"

using namespace Varjo;

Vector2::Vector2(float x_, float y_) : x(x_), y(y_)
{
}

namespace Varjo
{
	Vector2 operator+(const Vector2& v, const Vector2& w)
	{
		return Vector2(v.x + w.x, v.y + w.y);
	}

	Vector2 operator-(const Vector2& v, const Vector2& w)
	{
		return Vector2(v.x - w.x, v.y - w.y);
	}

	Vector2 operator*(const Vector2& v, const Vector2& w)
	{
		return Vector2(v.x * w.x, v.y * w.y);
	}

	Vector2 operator*(const Vector2& v, float s)
	{
		return Vector2(v.x * s, v.y * s);
	}

	Vector2 operator*(float s, const Vector2& v)
	{
		return Vector2(v.x * s, v.y * s);
	}

	Vector2 operator/(const Vector2& v, const Vector2& w)
	{
		return Vector2(v.x / w.x, v.y / w.y);
	}

	Vector2 operator/(const Vector2& v, float s)
	{
		float invS = 1.0f / s;
		return Vector2(v.x * invS, v.y * invS);
	}

	Vector2 operator-(const Vector2& v)
	{
		return Vector2(-v.x, -v.y);
	}

	bool operator==(const Vector2& v, const Vector2& w)
	{
		return MathUtils::almostSame(v.x, w.x) && MathUtils::almostSame(v.y, w.y);
	}

	bool operator!=(const Vector2& v, const Vector2& w)
	{
		return !(v == w);
	}

	bool operator>(const Vector2& v, const Vector2& w)
	{
		return v.x > w.x && v.y > w.y;
	}

	bool operator<(const Vector2& v, const Vector2& w)
	{
		return v.x < w.x && v.y < w.y;
	}
}

Vector2& Vector2::operator+=(const Vector2& v)
{
	*this = *this + v;
	return *this;
}

Vector2& Vector2::operator-=(const Vector2& v)
{
	*this = *this - v;
	return *this;
}

Vector2& Vector2::operator*=(const Vector2& v)
{
	*this = *this * v;
	return *this;
}

Vector2& Vector2::operator*=(float s)
{
	*this = *this * s;
	return *this;
}

Vector2& Vector2::operator/=(const Vector2& v)
{
	*this = *this / v;
	return *this;
}

Vector2& Vector2::operator/=(float s)
{
	*this = *this / s;
	return *this;
}

float Vector2::operator[](uint32_t index) const
{
	return (&x)[index];
}

float Vector2::getElement(uint32_t index) const
{
	switch (index)
	{
		case 0: return x;
		case 1: return y;
		default: return 0.0f;
	}
}

void Vector2::setElement(uint32_t index, float value)
{
	switch (index)
	{
		case 0: x = value;
		case 1: y = value;
		default: break;
	}
}

float Vector2::length() const
{
	return std::sqrt(x * x + y * y);
}

float Vector2::lengthSquared() const
{
	return (x * x + y * y);
}

void Vector2::normalize()
{
	*this /= length();
}

Vector2 Vector2::normalized() const
{
	return *this / length();
}

void Vector2::inverse()
{
	x = 1.0f / x;
	y = 1.0f / y;
}

Vector2 Vector2::inversed() const
{
	Vector2 inverse;

	inverse.x = 1.0f / x;
	inverse.y = 1.0f / y;

	return inverse;
}

bool Vector2::isZero() const
{
	return MathUtils::almostZero(x) && MathUtils::almostZero(y);
}

bool Vector2::isNan() const
{
	return (std::isnan(x) || std::isnan(y));
}

bool Vector2::isNormal() const
{
	return MathUtils::almostSame(lengthSquared(), 1.0f);
}

float Vector2::dot(const Vector2& v) const
{
	return (x * v.x) + (y * v.y);
}

Vector2 Vector2::reflect(const Vector2& normal) const
{
	return *this - ((2.0f * this->dot(normal)) * normal);
}

Vector2 Vector2::lerp(const Vector2& v1, const Vector2& v2, float t)
{
	assert(t >= 0.0f && t <= 1.0f);
	return v1 * (1.0f - t) + v2 * t;
}

Vector2 Vector2::abs(const Vector2& v)
{
	return Vector2(std::abs(v.x), std::abs(v.y));
}

std::string Vector2::toString() const
{
	return tfm::format("(%.2f, %.2f)", x, y);
}
