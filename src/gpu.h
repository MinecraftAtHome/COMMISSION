#pragma once

#include "common.h"
#include <optional>

struct SeedIterator {
    uint64_t start;
    std::atomic_uint64_t pos;
    std::optional<uint64_t> end;

    SeedIterator(uint64_t start, std::optional<uint64_t> end = {}) : start(start), pos(start), end(end) {

    }

    std::optional<uint64_t> next(uint64_t count) {
        uint64_t seed = pos.fetch_add(count);
        if (end && seed - start >= *end - start) return {};
        return seed;
    }

    bool exhausted() const {
        return end && pos.load(std::memory_order_relaxed) - start >= *end - start;
    }
};

struct GpuThread: Thread<GpuThread> {
    int device;
    SeedIterator &input;
    GpuOutputs &outputs;

    GpuThread(int device, SeedIterator &input, GpuOutputs &outputs);

    void run();
};