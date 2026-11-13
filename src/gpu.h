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

    // Whether a seed lies inside [start, end). The last batch of a range usually
    // overshoots end; results from that overshoot belong to the next range.
    bool contains(uint64_t seed) const {
        return !end || seed - start < *end - start;
    }
};

struct GpuThread: Thread<GpuThread> {
    int device;
    SeedIterator &input;
    GpuOutputs &outputs;
    // Every seed below done_pos has been fully processed by this GPU and its
    // candidates pushed to outputs. Written while holding outputs.mutex, so a
    // snapshot taken under that mutex is consistent with the queue contents.
    // Only meaningful for a single GPU thread (batches finish in order).
    std::atomic_uint64_t done_pos;

    GpuThread(int device, SeedIterator &input, GpuOutputs &outputs);

    void run();
};
