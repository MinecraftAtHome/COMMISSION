#pragma once

#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <array>

#ifndef OMISSION_LARGE_BIOMES
#define OMISSION_LARGE_BIOMES 0
#endif
#if OMISSION_LARGE_BIOMES
constexpr bool large_biomes = true;
#else
constexpr bool large_biomes = false;
#endif
#ifndef OMISSION_UNBOUND
#define OMISSION_UNBOUND 0
#endif
#if OMISSION_UNBOUND
constexpr bool unbound = true;
#else
constexpr bool unbound = false;
#endif

#ifndef PRINT_INTERVAL
#define PRINT_INTERVAL 256
#endif

constexpr std::array<char, 16> net_handshake { 'O', 'M', 'I', 'S', 'S', 'I', 'O', 'N', '-', 'G', 'P', 'U', ' ', large_biomes ? 'L' : 'S', 'B', '\n' };

struct GpuOutput {
    uint64_t seed;
    int32_t x;
    int32_t z;
};

struct CpuOutput {
    uint64_t seed;
    int32_t x;
    int32_t z;
    int32_t score;
};

struct GpuOutputs {
    std::queue<GpuOutput> queue;
    std::mutex mutex;
};

struct CpuOutputs {
    std::queue<CpuOutput> queue;
    std::mutex mutex;
};

struct HostService {
    std::string host;
    std::string service;
};

template<typename T>
struct Thread {
private:
    std::atomic_bool stop_flag;
    std::atomic_bool done_flag;
    std::thread thread;

protected:
    Thread() : stop_flag(false), done_flag(false), thread() {

    }

    void start() {
        thread = std::thread([this]() {
            static_cast<T*>(this)->run();
            done_flag.store(true, std::memory_order_release);
        });
    }

    bool should_stop() {
        return stop_flag.load(std::memory_order_relaxed);
    }

public:
    void stop() {
        stop_flag.store(true, std::memory_order_relaxed);
    }

    void join() {
        if (thread.joinable()) thread.join();
    }

    bool is_done() const {
        return done_flag.load(std::memory_order_acquire);
    }
};