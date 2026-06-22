#pragma once

#include "common.h"

struct CpuThread: Thread<CpuThread> {
    int id;
    int32_t min_size;
    GpuOutputs &inputs;
    CpuOutputs &outputs;

    CpuThread(int id, int32_t min_size, GpuOutputs &inputs, CpuOutputs &outputs);

    void run();
};