// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "tinyformat/tinyformat.h"

#include "Utils/Timer.h"

using namespace Varjo;
namespace sc = std::chrono;

std::string TimerData::getString(bool withMilliseconds) const
{
	if (withMilliseconds)
		return tfm::format("%02d:%02d:%02d.%03d", hours, minutes, seconds, milliseconds);
	else
		return tfm::format("%02d:%02d:%02d", hours, minutes, seconds);
}

Timer::Timer()
{
	restart();
}

void Timer::restart()
{
	startTime = sc::high_resolution_clock::now();
}

float Timer::getElapsedMilliseconds() const
{
	auto elapsedTime = sc::high_resolution_clock::now() - startTime;
	uint64_t totalNanoseconds = sc::duration_cast<sc::nanoseconds>(elapsedTime).count();
	return float(totalNanoseconds / 1000000.0);
}

float Timer::getElapsedSeconds() const
{
	auto elapsedTime = sc::high_resolution_clock::now() - startTime;
	uint64_t totalMilliseconds = sc::duration_cast<sc::milliseconds>(elapsedTime).count();
	return float(totalMilliseconds / 1000.0);
}

TimerData Timer::getElapsed() const
{
	auto elapsedTime = sc::high_resolution_clock::now() - startTime;

	TimerData timerData;

	timerData.totalNanoseconds = sc::duration_cast<sc::nanoseconds>(elapsedTime).count();
	timerData.totalMilliseconds = sc::duration_cast<sc::milliseconds>(elapsedTime).count();
	timerData.totalSeconds = sc::duration_cast<sc::seconds>(elapsedTime).count();
	timerData.totalMinutes = sc::duration_cast<sc::minutes>(elapsedTime).count();
	timerData.totalHours = sc::duration_cast<sc::hours>(elapsedTime).count();

	timerData.hours = timerData.totalHours;
	timerData.minutes = timerData.totalMinutes - timerData.hours * 60;
	timerData.seconds = timerData.totalSeconds - timerData.minutes * 60 - timerData.hours * 60 * 60;
	timerData.milliseconds = timerData.totalMilliseconds - timerData.seconds * 1000 - timerData.minutes * 60 * 1000 - timerData.hours * 60 * 60 * 1000;

	return timerData;
}
