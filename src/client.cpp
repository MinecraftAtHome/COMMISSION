#include "client.h"
#include <asio.hpp>

ClientThread::ClientThread(HostService &server_address, GpuOutputs &inputs) : server_address(server_address), inputs(inputs) {
    start();
}

void ClientThread::run() {
    std::printf("Started client thread\n");

    try {
        asio::io_context io_context;

        asio::ip::tcp::resolver resolver(io_context);
        auto endpoints = resolver.resolve(server_address.host, server_address.service);
        std::printf("Resolved\n");

        asio::ip::tcp::socket socket(io_context);
        asio::connect(socket, endpoints);
        auto remote_endpoint = socket.remote_endpoint();
        std::string address = remote_endpoint.address().to_string();
        std::printf("Connected to %s %d\n", address.c_str(), remote_endpoint.port());

        asio::write(socket, asio::buffer(net_handshake));

        std::vector<GpuOutput> pending_data;
        while (!should_stop()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            {
                std::lock_guard lock(inputs.mutex);
                while (!inputs.queue.empty()) {
                    pending_data.push_back(inputs.queue.front());
                    inputs.queue.pop();
                }
            }

            if (pending_data.empty()) continue;

            asio::write(socket, asio::buffer(pending_data));

            pending_data.clear();
        }
    } catch (std::exception &e) {
        std::fprintf(stderr, "Uncaught exception on client thread: %s\n", e.what());
        std::abort();
    }
}