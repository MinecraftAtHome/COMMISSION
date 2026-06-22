#include "server.h"
#include <asio.hpp>

class tcp_connection : public std::enable_shared_from_this<tcp_connection> {
public:
    typedef std::shared_ptr<tcp_connection> pointer;

    static pointer create(asio::io_context& io_context, GpuOutputs &outputs) {
        return pointer(new tcp_connection(io_context, outputs));
    }

    ~tcp_connection() {
        std::string address = remote_endpoint_.address().to_string();
        std::printf("Closing connection from %s %d\n", address.c_str(), remote_endpoint_.port());
    }

    asio::ip::tcp::socket& socket() {
        return socket_;
    }

    void start(asio::ip::tcp::endpoint remote_endpoint) {
        remote_endpoint_ = remote_endpoint;

        std::string address = remote_endpoint_.address().to_string();
        std::printf("Accepted connection from %s %d\n", address.c_str(), remote_endpoint_.port());

        asio::async_read(
            socket_,
            asio::buffer(handshake_),
            std::bind(&tcp_connection::handle_handshake, shared_from_this(), asio::placeholders::error, asio::placeholders::bytes_transferred)
        );
    }

private:
    tcp_connection(asio::io_context& io_context, GpuOutputs &outputs) : socket_(io_context), outputs_(outputs), remote_endpoint_(), pending_data_len_(0) {

    }

    void handle_handshake(const std::error_code& error, size_t bytes_transferred) {
        if (handshake_ != net_handshake) {
            std::string address = remote_endpoint_.address().to_string();
            std::printf("Received invalid handshake from %s %d\n", address.c_str(), remote_endpoint_.port());
        } else {
            start_read();
        }
    }

    void start_read() {
        socket_.async_read_some(
            asio::buffer(pending_data_) + pending_data_len_,
            std::bind(&tcp_connection::handle_read, shared_from_this(), asio::placeholders::error, asio::placeholders::bytes_transferred)
        );
    }

    void handle_read(const std::error_code& error, size_t bytes_transferred) {
        if (!error) {
            // std::printf("Read %zu\n", bytes_transferred);

            pending_data_len_ += bytes_transferred;

            if (pending_data_len_ >= sizeof(GpuOutput)) {
                size_t count = pending_data_len_ / sizeof(GpuOutput);
                // std::printf("Adding %zu items\n", count);
                {
                    std::lock_guard lock(outputs_.mutex);
                    for (size_t i = 0; i < count; i++) {
                        GpuOutput output;
                        std::memcpy(&output, pending_data_.data() + i * sizeof(GpuOutput), sizeof(GpuOutput));
                        outputs_.queue.push(output);
                    }
                }
                pending_data_len_ -= count * sizeof(GpuOutput);
                std::memmove(pending_data_.data(), pending_data_.data() + count * sizeof(GpuOutput), pending_data_len_);
            }

            start_read();
        }
    }

    asio::ip::tcp::socket socket_;
    GpuOutputs &outputs_;
    asio::ip::tcp::endpoint remote_endpoint_;
    std::array<char, sizeof(net_handshake)> handshake_;
    std::array<char, 1024> pending_data_;
    size_t pending_data_len_ = 0;
};

class tcp_server {
public:
    tcp_server(asio::io_context& io_context, asio::ip::tcp::endpoint endpoint, GpuOutputs &outputs_) : io_context_(io_context), acceptor_(io_context, endpoint), outputs_(outputs_) {
        std::string address = endpoint.address().to_string();
        std::printf("Listening on %s %d\n", address.c_str(), endpoint.port());

        start_accept();
    }

private:
    void start_accept() {
        tcp_connection::pointer new_connection = tcp_connection::create(io_context_, outputs_);

        acceptor_.async_accept(
            new_connection->socket(),
            accepted_remote_endpoint_,
            std::bind(&tcp_server::handle_accept, this, new_connection, asio::placeholders::error)
        );
    }

    void handle_accept(tcp_connection::pointer new_connection, const std::error_code& error) {
        if (!error) {
            new_connection->start(accepted_remote_endpoint_);
        }

        start_accept();
    }

    asio::io_context& io_context_;
    asio::ip::tcp::acceptor acceptor_;
    GpuOutputs &outputs_;
    asio::ip::tcp::endpoint accepted_remote_endpoint_;
};

ServerThread::ServerThread(HostService &listen_address, GpuOutputs &outputs) : listen_address(listen_address), outputs(outputs) {
    start();
}

void ServerThread::run() {
    std::printf("Started server thread\n");

    try {
        asio::io_context io_context;

        asio::ip::tcp::resolver resolver(io_context);
        auto endpoints = resolver.resolve(listen_address.host, listen_address.service);
        if (endpoints.empty()) {
            std::fprintf(stderr, "Resolution failed\n");
            std::abort();
        }
        auto endpoint = endpoints.begin()->endpoint();

        tcp_server server(io_context, endpoint, outputs);

        io_context.run();
    } catch (std::exception &e) {
        std::fprintf(stderr, "Uncaught exception on server thread: %s\n", e.what());
        std::abort();
    }
}