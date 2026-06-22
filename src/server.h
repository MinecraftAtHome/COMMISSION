#pragma once

#include "common.h"

struct ServerThread: Thread<ServerThread> {
    HostService &listen_address;
    GpuOutputs &outputs;

    ServerThread(HostService &listen_address, GpuOutputs &outputs);

    void run();
};