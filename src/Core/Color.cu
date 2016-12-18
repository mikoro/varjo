// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cassert>

#include "Core/Color.h"
#include "Math/MathUtils.h"
#include "Math/Vector3.h"

using namespace Varjo;

CUDA_CALLABLE Color::Color(float r_, float g_, float b_, float a_) : r(r_), g(g_), b(b_), a(a_)
{
}

CUDA_CALLABLE Color::Color(int32_t r_, int32_t g_, int32_t b_, int32_t a_)
{
	assert(r_ >= 0 && r_ <= 255 && g_ >= 0 && g_ <= 255 && b_ >= 0 && b_ <= 255 && a_ >= 0 && a_ <= 255);

	const float inv255 = 1.0f / 255.0f;

	r = r_ * inv255;
	g = g_ * inv255;
	b = b_ * inv255;
	a = a_ * inv255;
}

namespace Varjo
{
	CUDA_CALLABLE Color operator+(const Color& c1, const Color& c2)
	{
		return Color(c1.r + c2.r, c1.g + c2.g, c1.b + c2.b, c1.a + c2.a);
	}

	CUDA_CALLABLE Color operator-(const Color& c1, const Color& c2)
	{
		return Color(c1.r - c2.r, c1.g - c2.g, c1.b - c2.b, c1.a - c2.a);
	}

	CUDA_CALLABLE Color operator-(const Color& c)
	{
		return Color(-c.r, -c.g, -c.b, -c.a);
	}

	CUDA_CALLABLE Color operator*(const Color& c1, const Color& c2)
	{
		return Color(c1.r * c2.r, c1.g * c2.g, c1.b * c2.b, c1.a * c2.a);
	}

	CUDA_CALLABLE Color operator*(const Color& c, float s)
	{
		return Color(c.r * s, c.g * s, c.b * s, c.a * s);
	}

	CUDA_CALLABLE Color operator*(float s, const Color& c)
	{
		return Color(c.r * s, c.g * s, c.b * s, c.a * s);
	}

	CUDA_CALLABLE Color operator/(const Color& c1, const Color& c2)
	{
		return Color(c1.r / c2.r, c1.g / c2.g, c1.b / c2.b, c1.a / c2.a);
	}

	CUDA_CALLABLE Color operator/(const Color& c, float s)
	{
		float invS = 1.0f / s;
		return Color(c.r * invS, c.g * invS, c.b * invS, c.a * invS);
	}

	CUDA_CALLABLE bool operator==(const Color& c1, const Color& c2)
	{
		return MathUtils::almostSame(c1.r, c2.r) && MathUtils::almostSame(c1.g, c2.g) && MathUtils::almostSame(c1.b, c2.b) && MathUtils::almostSame(c1.a, c2.a);
	}

	CUDA_CALLABLE bool operator!=(const Color& c1, const Color& c2)
	{
		return !(c1 == c2);
	}
}

CUDA_CALLABLE Color& Color::operator+=(const Color& c)
{
	*this = *this + c;
	return *this;
}

CUDA_CALLABLE Color& Color::operator-=(const Color& c)
{
	*this = *this - c;
	return *this;
}

CUDA_CALLABLE Color& Color::operator*=(const Color& c)
{
	*this = *this * c;
	return *this;
}

CUDA_CALLABLE Color& Color::operator*=(float s)
{
	*this = *this * s;
	return *this;
}

CUDA_CALLABLE Color& Color::operator/=(float s)
{
	*this = *this / s;
	return *this;
}

CUDA_CALLABLE uint32_t Color::getRgbaValue() const
{
	assert(isClamped());

	uint32_t r_ = static_cast<uint32_t>(r * 255.0f + 0.5f) & 0xff;
	uint32_t g_ = static_cast<uint32_t>(g * 255.0f + 0.5f) & 0xff;
	uint32_t b_ = static_cast<uint32_t>(b * 255.0f + 0.5f) & 0xff;
	uint32_t a_ = static_cast<uint32_t>(a * 255.0f + 0.5f) & 0xff;

	return (r_ << 24 | g_ << 16 | b_ << 8 | a_);
}

CUDA_CALLABLE uint32_t Color::getAbgrValue() const
{
	assert(isClamped());

	uint32_t r_ = static_cast<uint32_t>(r * 255.0f + 0.5f) & 0xff;
	uint32_t g_ = static_cast<uint32_t>(g * 255.0f + 0.5f) & 0xff;
	uint32_t b_ = static_cast<uint32_t>(b * 255.0f + 0.5f) & 0xff;
	uint32_t a_ = static_cast<uint32_t>(a * 255.0f + 0.5f) & 0xff;

	return (a_ << 24 | b_ << 16 | g_ << 8 | r_);
}

CUDA_CALLABLE float Color::getLuminance() const
{
	return MAX(0.0f, 0.2126f * r + 0.7152f * g + 0.0722f * b); // expects linear space
}

CUDA_CALLABLE bool Color::isTransparent() const
{
	return (a < 1.0f);
}

CUDA_CALLABLE bool Color::isZero() const
{
	return (r == 0.0f && g == 0.0f && b == 0.0f); // ignore alpha
}

CUDA_CALLABLE bool Color::isClamped() const
{
	return (r >= 0.0f && r <= 1.0f && g >= 0.0f && g <= 1.0f && b >= 0.0f && b <= 1.0f && a >= 0.0f && a <= 1.0f);
}

CUDA_CALLABLE bool Color::isNan() const
{
	return (isnan(r) || isnan(g) || isnan(b) || isnan(a));
}

CUDA_CALLABLE bool Color::isNegative() const
{
	return (r < 0.0f || g < 0.0f || b < 0.0f || a < 0.0f);
}

CUDA_CALLABLE void Color::clamp()
{
	*this = clamped();
}

CUDA_CALLABLE Color Color::clamped() const
{
	Color c;

	c.r = MAX(0.0f, MIN(r, 1.0f));
	c.g = MAX(0.0f, MIN(g, 1.0f));
	c.b = MAX(0.0f, MIN(b, 1.0f));
	c.a = MAX(0.0f, MIN(a, 1.0f));

	return c;
}

CUDA_CALLABLE void Color::clampPositive()
{
	*this = clampedPositive();
}

CUDA_CALLABLE Color Color::clampedPositive() const
{
	Color c;

	c.r = (r < 0.0f) ? 0.0f : r;
	c.g = (g < 0.0f) ? 0.0f : g;
	c.b = (b < 0.0f) ? 0.0f : b;
	c.a = (a < 0.0f) ? 0.0f : a;

	return c;
}

CUDA_CALLABLE Color Color::fromRgbaValue(uint32_t rgba)
{
	const float inv255 = 1.0f / 255.0f;

	uint32_t r_ = (rgba >> 24);
	uint32_t g_ = (rgba >> 16) & 0xff;
	uint32_t b_ = (rgba >> 8) & 0xff;
	uint32_t a_ = rgba & 0xff;

	Color c;

	c.r = r_ * inv255;
	c.g = g_ * inv255;
	c.b = b_ * inv255;
	c.a = a_ * inv255;

	return c;
}

CUDA_CALLABLE Color Color::fromAbgrValue(uint32_t abgr)
{
	const float inv255 = 1.0f / 255.0f;

	uint32_t r_ = abgr & 0xff;
	uint32_t g_ = (abgr >> 8) & 0xff;
	uint32_t b_ = (abgr >> 16) & 0xff;
	uint32_t a_ = (abgr >> 24);

	Color c;

	c.r = r_ * inv255;
	c.g = g_ * inv255;
	c.b = b_ * inv255;
	c.a = a_ * inv255;

	return c;
}

CUDA_CALLABLE Color Color::fromNormal(const Vector3& normal)
{
	Color c;

	c.r = (normal.x + 1.0f) / 2.0f;
	c.g = (normal.y + 1.0f) / 2.0f;
	c.b = (normal.z + 1.0f) / 2.0f;

	return c;
}

CUDA_CALLABLE Color Color::fromFloat(float c)
{
	return Color(c, c, c, 1.0f);
}

CUDA_CALLABLE Color Color::lerp(const Color& start, const Color& end, float alpha)
{
	Color c;

	c.r = start.r + (end.r - start.r) * alpha;
	c.g = start.g + (end.g - start.g) * alpha;
	c.b = start.b + (end.b - start.b) * alpha;
	c.a = start.a + (end.a - start.a) * alpha;

	return c;
}

CUDA_CALLABLE Color Color::alphaBlend(const Color& first, const Color& second)
{
	const float alpha = second.a;
	const float invAlpha = 1.0f - alpha;

	Color c;

	c.r = (alpha * second.r + invAlpha * first.r);
	c.g = (alpha * second.g + invAlpha * first.g);
	c.b = (alpha * second.b + invAlpha * first.b);
	c.a = 1.0f;

	return c;
}

CUDA_CALLABLE Color Color::pow(const Color& color, float power)
{
	Color c;

	c.r = std::pow(color.r, power);
	c.g = std::pow(color.g, power);
	c.b = std::pow(color.b, power);
	c.a = std::pow(color.a, power);

	return c;
}

CUDA_CALLABLE Color Color::fastPow(const Color& color, float power)
{
	Color c;

	c.r = MathUtils::fastPow(color.r, power);
	c.g = MathUtils::fastPow(color.g, power);
	c.b = MathUtils::fastPow(color.b, power);
	c.a = MathUtils::fastPow(color.a, power);

	return c;
}

CUDA_CALLABLE Color Color::exp(const Color& color)
{
	Color c;

	c.r = std::exp(color.r);
	c.g = std::exp(color.g);
	c.b = std::exp(color.b);
	c.a = std::exp(color.a);

	return c;
}

CUDA_CALLABLE Color Color::red()
{
	return Color(1.0f, 0.0f, 0.0f);;
}

CUDA_CALLABLE Color Color::green()
{
	return Color(0.0f, 1.0f, 0.0f);;
}

CUDA_CALLABLE Color Color::blue()
{
	return Color(0.0f, 0.0f, 1.0f);;
}

CUDA_CALLABLE Color Color::white()
{
	return Color(1.0f, 1.0f, 1.0f);;
}

CUDA_CALLABLE Color Color::black()
{
	return Color(0.0f, 0.0f, 0.0f);;
}
