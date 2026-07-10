#pragma once

#include "common.h"

namespace asio { class io_context; }

struct ServerThread: Thread<ServerThread> {
    HostService &listen_address;
    GpuOutputs &outputs;
    std::atomic<asio::io_context*> io_ctx{nullptr};

    ServerThread(HostService &listen_address, GpuOutputs &outputs);

    void run();
    void shutdown();
};