#include "common.h"
#ifndef NO_GPU
#include "gpu.h"
#endif
#ifndef NO_CPU
#include "cpu.h"
#endif
#ifndef NO_NET
#include "client.h"
#include "server.h"
#endif

#include <cstdint>
#include <cstring>
#include <cinttypes>
#include <cstdio>
#include <chrono>
#include <optional>
#include <charconv>
#include <algorithm>
#include <random>
#include <atomic>

#include <fstream>
#include <sstream>
#include <iomanip>

#ifdef NO_GPU
constexpr bool no_gpu = true;
#else
constexpr bool no_gpu = false;
#endif
#ifdef NO_CPU
constexpr bool no_cpu = true;
#else
constexpr bool no_cpu = false;
#endif
#ifdef NO_NET
constexpr bool no_net = true;
#else
constexpr bool no_net = false;
#endif

std::optional<HostService> split_address(std::string_view address) {
    size_t i = address.find_last_of(':');
    if (i == std::string_view::npos) return {};

    return {{ std::string(address.substr(0, i)), std::string(address.substr(i + 1)) }};
}

bool check_duplicate(bool duplicate, const char *option) {
    if (duplicate) {
        std::fprintf(stderr, "duplicate %s option\n", option);
        return true;
    }
    return false;
}

bool check_argument(int argc, int i, const char *option) {
    if (i >= argc) {
        std::fprintf(stderr, "missing argument to %s\n", option);
        return true;
    }
    return false;
}

template<typename T, typename F>
bool parse_argument_int(int argc, const char *const *argv, int &i, std::optional<T> &out, F &&test, const char *option) {
    if (check_duplicate((bool)out, option)) return false;
    if (check_argument(argc, i, argv[i - 1])) return false;
    const char *arg_val = argv[i++];
    const char *arg_val_end = arg_val + std::strlen(arg_val);
    T val;
    auto [ptr, ec] = std::from_chars(arg_val, arg_val_end, val);
    if (ec != std::errc() || ptr != arg_val_end || !test(val)) {
        std::fprintf(stderr, "invalid argument to %s: %s\n", option, arg_val);
        return false;
    }
    out = val;
    return true;
}

struct Args {
    std::vector<int> devices;
    std::optional<int> threads;
    std::optional<HostService> client;
    std::optional<HostService> server;
    std::optional<std::string> output_file;
    std::optional<int64_t> start_seed;

    std::optional<int32_t> min_size;

    bool parse(int argc, const char **const argv) {
        for (int i = 1; i < argc;) {
            const char *arg = argv[i++];

            if (std::strcmp("--device", arg) == 0) {
                if (check_argument(argc, i, arg)) return false;
                const char *devices_str = argv[i++];
                const char *last = devices_str + std::strlen(devices_str);
                const char *first = devices_str;
                while (first != last) {
                    int device;
                    auto [ptr, ec] = std::from_chars(first, last, device, 10);
                    if (ec != std::errc() || device < 0 || std::find(devices.begin(), devices.end(), device) != devices.end() || ptr != last && *ptr != ',') {
                        std::fprintf(stderr, "invalid argument to --device: %s\n", devices_str);
                        return false;
                    }
                    devices.push_back(device);
                    first = ptr;
                    if (first != last) first++;
                }
            } else if (std::strcmp("--threads", arg) == 0) {
                if (!parse_argument_int(argc, argv, i, threads, [](int threads){ return threads >= 1 && threads <= 1024; }, arg)) return false;
            } else if (std::strcmp("--client", arg) == 0) {
                if (check_duplicate((bool)client, arg)) return false;
                if (check_argument(argc, i, arg)) return false;
                auto address = split_address(argv[i++]);
                if (!address) {
                    std::fprintf(stderr, "invalid argument to --client\n");
                    return false;
                }
                client = std::move(address);
            } else if (std::strcmp("--server", arg) == 0) {
                if (check_duplicate((bool)server, arg)) return false;
                if (check_argument(argc, i, arg)) return false;
                auto address = split_address(argv[i++]);
                if (!address) {
                    std::fprintf(stderr, "invalid argument to --server\n");
                    return false;
                }
                server = std::move(address);
            } else if (std::strcmp("--output", arg) == 0) {
                if (check_duplicate((bool)output_file, arg)) return false;
                if (check_argument(argc, i, arg)) return false;
                output_file = argv[i++];
            } else if (std::strcmp("--start", arg) == 0) {
                if (!parse_argument_int(argc, argv, i, start_seed, [](int64_t start_seed){ return true; }, arg)) return false;

            } else if (std::strcmp("--size", arg) == 0) {
                if (!parse_argument_int(argc, argv, i, min_size, [](int32_t min_size){ return min_size >= 0; }, arg)) return false;
            } else {
                std::fprintf(stderr, "unknown option: %s\n", arg);
                return false;
            }
        }

        if (threads && client) {
            std::fprintf(stderr, "--threads and --client are mutually exclusive\n");
            return false;
        }

        if (output_file && client) {
            std::fprintf(stderr, "--output and --client are mutually exclusive\n");
            return false;
        }

        if (devices.empty() && !server) {
            devices.push_back(0);
        }

        if (start_seed && devices.empty()) {
            std::fprintf(stderr, "--start does nothing when not running gpus\n");
            return false;
        }

        if (min_size && !threads && client) {
            std::fprintf(stderr, "--size does nothing when not running cpu threads\n");
            return false;
        }

        return true;
    }
};

uint64_t random_start_seed() {
    std::random_device device;
    return ((uint64_t)device() << 32) + (uint64_t)device();
}

namespace {


struct SearchResult {
    int64_t seed = 0;
    int32_t x = 0;
    int32_t z = 0;
    int32_t size = 0;
};

std::string join_devices(const std::vector<int> &devices) {
    if (devices.empty()) {
        return "(none)";
    }

    std::string result;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(devices[i]);
    }
    return result;
}

std::string platform_name(bool large_biomes) {
    return large_biomes ? "java_1_21_5_lb" : "java_1_21_5";
}

std::string make_chunkbase_url(int64_t seed, int32_t x, int32_t z, bool large_biomes) {
    return "https://www.chunkbase.com/apps/seed-map#seed=" + std::to_string(seed) +
           "&platform=" + platform_name(large_biomes) +
           "&dimension=overworld&x=" + std::to_string(x) +
           "&z=" + std::to_string(z) +
           "&zoom=0.125";
}

std::string make_mcseedmap_url(int64_t seed, int32_t x, int32_t z, bool large_biomes) {
    const std::string base = large_biomes
        ? "https://mcseedmap.net/1.21.5-Java/lb/"
        : "https://mcseedmap.net/1.21.5-Java/";
    return base + std::to_string(seed) +
           "#x=" + std::to_string(x) +
           "&z=" + std::to_string(z) +
           "&l=-3";
}

std::string current_finding_tag() {
    if (unbound) {
        return large_biomes ? "ULB" : "USB";
    }
    return large_biomes ? "LB" : "SB";
}



void write_result_line(std::FILE *file, const SearchResult &result) {
    const std::string tag = current_finding_tag();
    std::fprintf(file, "%s %" PRIi64 " %" PRIi32 " %" PRIi32 " %" PRIi32 "\n",
                 tag.c_str(), result.seed, result.x, result.z, result.size);
}

void print_result_card(const SearchResult &r, bool large_biomes, size_t rank = 0) {
    if (rank != 0) {
        std::printf("%2zu) ", rank);
    } else {
        std::printf("Found: ");
    }
    std::printf("seed=%" PRIi64 "  x=%" PRIi32 "  z=%" PRIi32 "  size=%" PRIi32 "\n",
                r.seed, r.x, r.z, r.size);
    std::printf("   Chunkbase : %s\n", make_chunkbase_url(r.seed, r.x, r.z, large_biomes).c_str());
    std::printf("   mcseedmap : %s\n", make_mcseedmap_url(r.seed, r.x, r.z, large_biomes).c_str());
}



void print_banner(const Args &args, int threads, int32_t min_size, const std::string &output_path, uint64_t start_seed) {
    std::printf("============================================================\n");
    std::printf("Minecraft Mushroom Island Search\n");
    std::printf("============================================================\n");
    std::printf("large_biomes    : %s\n", large_biomes ? "true" : "false");
    std::printf("unbound         : %s\n", unbound ? "true" : "false");
    std::printf("print interval  : %d\n", PRINT_INTERVAL);
    std::printf("GPU devices     : %s\n", join_devices(args.devices).c_str());
    std::printf("finding type    : %s\n", current_finding_tag().c_str());
    std::printf("CPU threads     : %d\n", threads);
    std::printf("min size        : %" PRIi32 "\n", min_size);
    std::printf("output file     : %s\n", output_path.c_str());
    std::printf("start seed      : %" PRIu64 "\n", start_seed);

    std::printf("============================================================\n\n");
}

}  // namespace

int main_inner(int argc, char **argv) {
    Args args{};
    if (!args.parse(argc, const_cast<const char **const>(argv))) {
        std::fprintf(stderr, "Usage:\n%s [--device <device>,<device>,...] [--threads <threads>] [--client <server_address>] [--server <listen_address>] [--output <output_file>] [--start <start_seed>] [--size <min_size>]\n", argv[0]);
        return 1;
    }

    const int threads = args.threads.value_or(args.client ? 0 : 1);

    if (no_gpu && args.devices.size() != 0) {
        std::fprintf(stderr, "The program was compiled without gpu support\n");
        return 1;
    }
    if (no_cpu && threads != 0) {
        std::fprintf(stderr, "The program was compiled without cpu support\n");
        return 1;
    }
    if (no_net && (args.client || args.server)) {
        std::fprintf(stderr, "The program was compiled without net support\n");
        return 1;
    }



    int32_t min_size = args.min_size.value_or(6'000'000 * (large_biomes ? 16 : 1));
    const std::string output_file_path = args.output_file ? args.output_file.value() : "output.txt";
    const uint64_t start_seed = args.start_seed.value_or(random_start_seed());

    std::FILE *output_file = nullptr;
    if (threads != 0) {
        output_file = std::fopen(output_file_path.c_str(), "a");
        if (output_file == nullptr) {
            std::fprintf(stderr, "Could not open %s\n", output_file_path.c_str());
            return 1;
        }
        std::setvbuf(output_file, nullptr, _IOLBF, BUFSIZ);
        std::fprintf(output_file, "\n");
        std::fflush(output_file);
    }

    print_banner(args, threads, min_size, output_file_path, start_seed);

    GpuOutputs gpu_outputs;
    CpuOutputs cpu_outputs;

#ifndef NO_GPU
    SeedIterator seed_range(start_seed);

    std::vector<std::unique_ptr<GpuThread>> gpu_threads;
    for (int device : args.devices) {
        gpu_threads.emplace_back(std::make_unique<GpuThread>(device, std::ref(seed_range), std::ref(gpu_outputs)));
    }
#endif

#ifndef NO_CPU
    std::vector<std::unique_ptr<CpuThread>> cpu_threads;
    for (int i = 0; i < threads; i++) {
        cpu_threads.emplace_back(std::make_unique<CpuThread>(i, min_size, std::ref(gpu_outputs), std::ref(cpu_outputs)));
    }
#endif

#ifndef NO_NET
    std::unique_ptr<ClientThread> client_thread;
    if (args.client) {
        client_thread = std::make_unique<ClientThread>(args.client.value(), std::ref(gpu_outputs));
    }

    std::unique_ptr<ServerThread> server_thread;
    if (args.server) {
        server_thread = std::make_unique<ServerThread>(args.server.value(), std::ref(gpu_outputs));
    }
#endif

    for (size_t i = 0;; ++i) {
        if (threads != 0) {
            std::lock_guard lock(cpu_outputs.mutex);
            while (!cpu_outputs.queue.empty()) {
                auto output = cpu_outputs.queue.front();
                cpu_outputs.queue.pop();
                print_result_card(SearchResult{static_cast<int64_t>(output.seed), output.x, output.z, output.score}, large_biomes);
                if (output_file != nullptr) {
                    write_result_line(output_file, SearchResult{static_cast<int64_t>(output.seed), output.x, output.z, output.score});
                    std::fflush(output_file);
                }
            }
        }

        if (args.devices.size() == 0 && i % 10 == 0) {
            std::lock_guard lock(gpu_outputs.mutex);
            std::printf("[status] queued GPU outputs: %zu\n", gpu_outputs.queue.size());
        }



        std::this_thread::sleep_for(std::chrono::seconds(1));
    }



#ifndef NO_GPU
    for (auto &thread : gpu_threads) {
        (*thread).stop();
    }
#endif
#ifndef NO_CPU
    for (auto &thread : cpu_threads) {
        (*thread).stop();
    }
#endif
#ifndef NO_NET
    if (client_thread) {
        (*client_thread).stop();
    }
    if (server_thread) {
        (*server_thread).stop();
    }
#endif

#ifndef NO_GPU
    for (auto &thread : gpu_threads) {
        (*thread).join();
    }
#endif
#ifndef NO_CPU
    for (auto &thread : cpu_threads) {
        (*thread).join();
    }
#endif
#ifndef NO_NET
    if (client_thread) {
        (*client_thread).join();
    }
    if (server_thread) {
        (*server_thread).join();
    }
#endif

    if (threads != 0) {
        std::lock_guard lock(cpu_outputs.mutex);
        while (!cpu_outputs.queue.empty()) {
            auto output = cpu_outputs.queue.front();
            cpu_outputs.queue.pop();
            print_result_card(SearchResult{static_cast<int64_t>(output.seed), output.x, output.z, output.score}, large_biomes);
            if (output_file != nullptr) {
                write_result_line(output_file, SearchResult{static_cast<int64_t>(output.seed), output.x, output.z, output.score});
                std::fflush(output_file);
            }
        }
    }

    if (output_file != nullptr) {
        std::fflush(output_file);
        std::fclose(output_file);
    }


    return 0;
}

int main(int argc, char **argv) {
    try {
        return main_inner(argc, argv);
    } catch (std::exception &e) {
        std::fprintf(stderr, "Uncaught exception in main: %s\n", e.what());
        std::abort();
    }
}
