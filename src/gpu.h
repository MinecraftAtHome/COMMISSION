#pragma once

#include "common.h"

struct SeedIterator {
    std::atomic_uint64_t pos;

    SeedIterator(uint64_t start) : pos(start) {

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