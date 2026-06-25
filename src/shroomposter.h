#pragma once

#include <cstdint>

struct PostResult
{
    int64_t seed;
    int32_t x;
    int32_t z;
    int64_t claimed_size;
};

void shroomposter_start();

void shroomposter_stop();

void shroomposter_submit(
    int64_t seed,
    int32_t x,
    int32_t z,
    int64_t claimed_size
);