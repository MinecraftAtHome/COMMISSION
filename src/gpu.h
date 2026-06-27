#pragma once

#include "common.h"
#include <optional>

struct SeedIterator {
    std::atomic_uint64_t pos;
    std::optional<uint64_t> stop_seed;

    SeedIterator(uint64_t start, std::optional<uint64_t> stop = std::nullopt) : pos(start), stop_seed(stop) {

    }

    uint64_t next(uint64_t count) {
        return pos.fetch_add(count);
    }
};

struct GpuThread: Thread<GpuThread> {
    int device;
    SeedIterator &input;
    GpuOutputs &outputs;

    GpuThread(int device, SeedIterator &input, GpuOutputs &outputs);

    void run();
};