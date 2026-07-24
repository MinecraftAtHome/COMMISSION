#pragma once

#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <array>
#include <unordered_set>

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

inline bool operator==(const CpuOutput &a, const CpuOutput &b) {
    return a.seed == b.seed && a.x == b.x && a.z == b.z && a.score == b.score;
}

struct CpuOutputHash {
    size_t operator()(const CpuOutput &output) const {
        size_t h = std::hash<uint64_t>{}(output.seed);
        h ^= std::hash<int32_t>{}(output.x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(output.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(output.score) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct GpuOutputs {
    std::queue<GpuOutput> queue;
    std::mutex mutex;
};

struct CpuOutputs {
    std::queue<CpuOutput> queue;
    std::unordered_set<CpuOutput, CpuOutputHash> seen;
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
    std::thread thread;

protected:
    Thread() : stop_flag(false), thread() {

    }

    void start() {
        thread = std::thread(&T::run, (T*)this);
    }

    bool should_stop() {
        return stop_flag.load(std::memory_order_relaxed);
    }

public:
    void stop() {
        stop_flag.store(true, std::memory_order_relaxed);
    }

    void join() {
        thread.join();
    }
};
