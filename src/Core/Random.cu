// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Color.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Core/Random.h"

#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4146)
#endif

using namespace Varjo;

CUDA_CALLABLE RandomGenerator::RandomGenerator()
{
	state.state = 0x853c49e6748fea9bULL;
	state.inc = 0xda3e39cb94b95bdbULL;
}

CUDA_CALLABLE RandomGenerator::RandomGenerator(uint64_t seed_)
{
	seed(seed_);
}

CUDA_CALLABLE RandomGenerator::RandomGenerator(RandomGeneratorState state_)
{
	seed(state_);
}

CUDA_CALLABLE void RandomGenerator::seed(uint64_t seed_)
{
	state.state = seed_;
	state.inc = reinterpret_cast<uint64_t>(this);
}

CUDA_CALLABLE void RandomGenerator::seed(RandomGeneratorState state_)
{
	state = state_;
}

// https://github.com/imneme/pcg-c-basic/blob/master/pcg_basic.c
CUDA_CALLABLE RandomGenerator::result_type RandomGenerator::operator()()
{
	uint64_t oldstate = state.state;
	state.state = oldstate * 6364136223846793005ULL + state.inc;
	uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
	uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);

	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

CUDA_CALLABLE Random::Random()
{
}

CUDA_CALLABLE Random::Random(uint64_t seed_)
{
	generator.seed(seed_);
}

CUDA_CALLABLE Random::Random(RandomGeneratorState state_)
{
	generator.seed(state_);
}

CUDA_CALLABLE void Random::seed(uint64_t seed_)
{
	generator.seed(seed_);
}

CUDA_CALLABLE void Random::seed(RandomGeneratorState state_)
{
	generator.seed(state_);
}

CUDA_CALLABLE RandomGeneratorState Random::getState() const
{
	return generator.state;
}

CUDA_CALLABLE uint32_t Random::getUint32()
{
	return generator();
}

CUDA_CALLABLE uint32_t Random::getUint32(uint32_t max)
{
	uint32_t threshold = -max % max;

	for (;;)
	{
		uint32_t value = generator();

		if (value >= threshold)
			return value % max;
	}
}

CUDA_CALLABLE uint32_t Random::getUint32(uint32_t min, uint32_t max)
{
	return getUint32((max - min) + 1) + min;
}

// http://mumble.net/~campbell/tmp/random_real.c
CUDA_CALLABLE float Random::getFloat()
{
#ifdef __CUDA_ARCH__
	return float(generator()) / float(0xFFFFFFFF);
#else
	return float(ldexp(generator(), -32));
#endif
}

CUDA_CALLABLE Color Random::getColor(bool randomAlpha)
{
	Color c;

	c.r = getFloat();
	c.g = getFloat();
	c.b = getFloat();
	c.a = randomAlpha ? getFloat() : 1.0f;

	return c;
}

CUDA_CALLABLE Vector2 Random::getVector2()
{
	Vector2 v;

	v.x = getFloat();
	v.y = getFloat();

	return v;
}

CUDA_CALLABLE Vector3 Random::getVector3()
{
	Vector3 v;

	v.x = getFloat();
	v.y = getFloat();
	v.z = getFloat();

	return v;
}

#ifdef _MSC_VER
#pragma warning (pop)
#endif
