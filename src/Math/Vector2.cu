// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "tinyformat/tinyformat.h"

#include "Math/Vector2.h"
#include "Math/MathUtils.h"

using namespace Varjo;

CUDA_CALLABLE Vector2::Vector2(float x_, float y_) : x(x_), y(y_)
{
}

namespace Varjo
{
	CUDA_CALLABLE Vector2 operator+(const Vector2& v, const Vector2& w)
	{
		return Vector2(v.x + w.x, v.y + w.y);
	}

	CUDA_CALLABLE Vector2 operator-(const Vector2& v, const Vector2& w)
	{
		return Vector2(v.x - w.x, v.y - w.y);
	}

	CUDA_CALLABLE Vector2 operator*(const Vector2& v, const Vector2& w)
	{
		return Vector2(v.x * w.x, v.y * w.y);
	}

	CUDA_CALLABLE Vector2 operator*(const Vector2& v, float s)
	{
		return Vector2(v.x * s, v.y * s);
	}

	CUDA_CALLABLE Vector2 operator*(float s, const Vector2& v)
	{
		return Vector2(v.x * s, v.y * s);
	}

	CUDA_CALLABLE Vector2 operator/(const Vector2& v, const Vector2& w)
	{
		return Vector2(v.x / w.x, v.y / w.y);
	}

	CUDA_CALLABLE Vector2 operator/(const Vector2& v, float s)
	{
		float invS = 1.0f / s;
		return Vector2(v.x * invS, v.y * invS);
	}

	CUDA_CALLABLE Vector2 operator-(const Vector2& v)
	{
		return Vector2(-v.x, -v.y);
	}

	CUDA_CALLABLE bool operator==(const Vector2& v, const Vector2& w)
	{
		return MathUtils::almostSame(v.x, w.x) && MathUtils::almostSame(v.y, w.y);
	}

	CUDA_CALLABLE bool operator!=(const Vector2& v, const Vector2& w)
	{
		return !(v == w);
	}

	CUDA_CALLABLE bool operator>(const Vector2& v, const Vector2& w)
	{
		return v.x > w.x && v.y > w.y;
	}

	CUDA_CALLABLE bool operator<(const Vector2& v, const Vector2& w)
	{
		return v.x < w.x && v.y < w.y;
	}
}

CUDA_CALLABLE Vector2& Vector2::operator+=(const Vector2& v)
{
	*this = *this + v;
	return *this;
}

CUDA_CALLABLE Vector2& Vector2::operator-=(const Vector2& v)
{
	*this = *this - v;
	return *this;
}

CUDA_CALLABLE Vector2& Vector2::operator*=(const Vector2& v)
{
	*this = *this * v;
	return *this;
}

CUDA_CALLABLE Vector2& Vector2::operator*=(float s)
{
	*this = *this * s;
	return *this;
}

CUDA_CALLABLE Vector2& Vector2::operator/=(const Vector2& v)
{
	*this = *this / v;
	return *this;
}

CUDA_CALLABLE Vector2& Vector2::operator/=(float s)
{
	*this = *this / s;
	return *this;
}

CUDA_CALLABLE float Vector2::operator[](uint32_t index) const
{
	return (&x)[index];
}

CUDA_CALLABLE float Vector2::getElement(uint32_t index) const
{
	switch (index)
	{
		case 0: return x;
		case 1: return y;
		default: return 0.0f;
	}
}

CUDA_CALLABLE void Vector2::setElement(uint32_t index, float value)
{
	switch (index)
	{
		case 0: x = value;
		case 1: y = value;
		default: break;
	}
}

CUDA_CALLABLE float Vector2::length() const
{
	return std::sqrt(x * x + y * y);
}

CUDA_CALLABLE float Vector2::lengthSquared() const
{
	return (x * x + y * y);
}

CUDA_CALLABLE void Vector2::normalize()
{
	*this /= length();
}

CUDA_CALLABLE Vector2 Vector2::normalized() const
{
	return *this / length();
}

CUDA_CALLABLE void Vector2::inverse()
{
	x = 1.0f / x;
	y = 1.0f / y;
}

CUDA_CALLABLE Vector2 Vector2::inversed() const
{
	Vector2 inverse;

	inverse.x = 1.0f / x;
	inverse.y = 1.0f / y;

	return inverse;
}

CUDA_CALLABLE bool Vector2::isZero() const
{
	return MathUtils::almostZero(x) && MathUtils::almostZero(y);
}

bool Vector2::isNan() const
{
	return (std::isnan(x) || std::isnan(y));
}

CUDA_CALLABLE bool Vector2::isNormal() const
{
	return MathUtils::almostSame(lengthSquared(), 1.0f);
}

CUDA_CALLABLE float Vector2::dot(const Vector2& v) const
{
	return (x * v.x) + (y * v.y);
}

CUDA_CALLABLE Vector2 Vector2::reflect(const Vector2& normal) const
{
	return *this - ((2.0f * this->dot(normal)) * normal);
}

CUDA_CALLABLE Vector2 Vector2::lerp(const Vector2& v1, const Vector2& v2, float t)
{
	assert(t >= 0.0f && t <= 1.0f);
	return v1 * (1.0f - t) + v2 * t;
}

CUDA_CALLABLE Vector2 Vector2::abs(const Vector2& v)
{
	return Vector2(std::abs(v.x), std::abs(v.y));
}

std::string Vector2::toString() const
{
	return tfm::format("(%.2f, %.2f)", x, y);
}
