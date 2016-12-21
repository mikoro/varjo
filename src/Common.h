// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#ifdef _MSC_VER
#define ALIGN(x) __declspec(align(x))
#elif __GNUC__
#define ALIGN(x) __attribute__ ((aligned(x)))
#endif

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
