#pragma once

#include "common.h"

struct ClientThread: Thread<ClientThread> {
    HostService &server_address;
    GpuOutputs &inputs;

    ClientThread(HostService &server_address, GpuOutputs &inputs);

    void run();
};