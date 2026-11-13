#pragma once

#include "common.h"

#include <optional>

struct CpuThread: Thread<CpuThread> {
    int id;
    int32_t min_size;
    GpuOutputs &inputs;
    CpuOutputs &outputs;
    // The input this thread is currently testing, guarded by inputs.mutex.
    // Together with inputs.queue it forms the set of not-yet-finished
    // candidates, which is what a checkpoint has to save.
    std::optional<GpuOutput> current;

    CpuThread(int id, int32_t min_size, GpuOutputs &inputs, CpuOutputs &outputs);

    void run();
};