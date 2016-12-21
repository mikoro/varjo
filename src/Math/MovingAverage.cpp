// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cassert>

#include "Math/MovingAverage.h"

using namespace Varjo;

MovingAverage::MovingAverage(float alpha_, float average_)
{
	setAlpha(alpha_);
	setAverage(average_);
}

void MovingAverage::setAlpha(float alpha_)
{
	assert(alpha_ > 0.0f && alpha_ <= 1.0f);

	alpha = alpha_;
}

void MovingAverage::setAverage(float average_)
{
	average = average_;
}

void MovingAverage::addMeasurement(float value)
{
	average = alpha * value + (1.0f - alpha) * average;
}

float MovingAverage::getAverage() const
{
	return average;
}
