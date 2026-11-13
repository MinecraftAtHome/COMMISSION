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
#ifdef BOINC
#include "boinc_glue.h"
#endif

#include <cstdint>
#include <cstring>
#include <cinttypes>
#include <cstdio>
#include <csignal>
#include <chrono>
#include <optional>
#include <charconv>
#include <algorithm>
#include <random>

#ifdef __linux__
#include <unistd.h>
#elif _WIN32
#include <cstdlib>
#endif
static std::atomic_bool running{true};
static std::atomic_uint32_t signal_count{0};

static void signal_handler(int) {
    uint32_t count = signal_count.fetch_add(1, std::memory_order_relaxed) + 1;
    running.store(false, std::memory_order_relaxed);
    if (count >= 2) {
#ifdef __linux__
        _exit(130);
#elif _WIN32
        std::_Exit(130);
#endif
    }
}

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
    std::optional<int64_t> end_seed;
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
            } else if (std::strcmp("--end", arg) == 0) {
                if (!parse_argument_int(argc, argv, i, end_seed, [](int64_t end_seed){ return true; }, arg)) return false;
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

        if (end_seed && !start_seed) {
            std::fprintf(stderr, "--end requires --start\n");
            return false;
        }

        if (end_seed && end_seed.value() == start_seed.value()) {
            std::fprintf(stderr, "--end must not equal --start\n");
            return false;
        }

        if (min_size && !threads && client) {
            std::fprintf(stderr, "--size does nothing when not running cpu threads\n");
            return false;
        }

#ifdef BOINC
        if (!start_seed || !end_seed) {
            std::fprintf(stderr, "the BOINC build needs --start and --end (the workunit's seed range)\n");
            return false;
        }

        if (client || server) {
            std::fprintf(stderr, "--client and --server are not supported in the BOINC build\n");
            return false;
        }
#endif

        return true;
    }
};

uint64_t random_start_seed() {
    std::random_device device;
    return ((uint64_t)device() << 32) + (uint64_t)device();
}


static void print_result(std::FILE *stream, const CpuOutput &output) {
    std::fprintf(stream, "%" PRIi64 " at %" PRIi32 " %" PRIi32 " with %" PRIi32 "\n", output.seed, output.x, output.z, output.score);
}

int main_inner(int argc, char **argv) {
#ifdef BOINC
    // Before anything else: sets up the client connection and redirects stderr
    // to stderr.txt (which is uploaded with the result).
    boinc_glue::init();
#endif

    Args args{};
    if (!args.parse(argc, const_cast<const char **const>(argv))) {
        std::fprintf(stderr, "Usage:\n%s [--device <device>,<device>,...] [--threads <threads>] [--client <server_address>] [--server <listen_address>] [--output <output_file>] [--start <start_seed>] [--end <end_seed>] [--size <min_size>]\n", argv[0]);
        return 1;
    }

#ifdef BOINC
    // The client tells us which GPU to use; it may also append "--device N" to
    // the command line, but the init data is the authoritative source.
    if (int device = boinc_glue::assigned_gpu_device(); device >= 0) {
        args.devices = { device };
    }
    if (args.devices.size() != 1) {
        std::fprintf(stderr, "the BOINC build runs exactly one GPU per task\n");
        return 1;
    }
#endif

    const int threads = args.threads.value_or(args.client ? 0 : 1);
    int32_t min_size = args.min_size.value_or(10'000'000 * (large_biomes ? 16 : 1));
    if (threads != 0) {
        std::printf("min_size = %" PRIi32 "\n", min_size);
    }

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

    std::printf("Hello! large_biomes = %s\nunbound: %s\nprint interval: %d\n", large_biomes ? "true" : "false", unbound ? "true" : "false", PRINT_INTERVAL);

#ifdef BOINC
    // Results are kept in memory (and in the checkpoint) and written once at
    // the end, so a restart can never leave duplicate or partial lines behind.
    const std::string output_path = boinc_glue::resolve_filename(args.output_file.value_or("out.txt"));

    boinc_glue::Checkpoint progress;
    progress.start = (uint64_t)args.start_seed.value();
    progress.end = (uint64_t)args.end_seed.value();
    progress.min_size = min_size;
    progress.large_biomes = large_biomes;
    progress.unbound = unbound;
    progress.pos = progress.start;
    {
        boinc_glue::Checkpoint saved;
        if (boinc_glue::load_checkpoint(saved)) {
            if (saved.same_workunit(progress)) {
                progress = std::move(saved);
                std::fprintf(stderr, "Resuming from checkpoint: pos %" PRIi64 ", %zu results, %zu pending candidates\n",
                             (int64_t)progress.pos, progress.results.size(), progress.pending.size());
            } else {
                std::fprintf(stderr, "Ignoring checkpoint from a different workunit\n");
            }
        }
    }
    std::fprintf(stderr, "COMMISSION %s %s: seeds %" PRIi64 " to %" PRIi64 " on device %d, min_size %" PRIi32 "\n",
                 large_biomes ? "lb" : "sb", unbound ? "unbound" : "bounded",
                 (int64_t)progress.start, (int64_t)progress.end, args.devices[0], min_size);
#else
    std::FILE *output_file = nullptr;
    if (threads != 0) {
        const char *output_file_path = args.output_file ? args.output_file.value().c_str() : "output.txt";
        output_file = std::fopen(output_file_path, "a");
        if (output_file == nullptr) {
            std::fprintf(stderr, "Could not open %s\n", output_file_path);
            return 1;
        }
        std::fprintf(output_file, "\n");
        std::fflush(output_file);
    }
#endif

    // Results are drained from the CPU threads by the main thread only.
    auto handle_result = [&](const CpuOutput &output) {
        print_result(stdout, output);
#ifdef BOINC
        print_result(stderr, output);
        progress.results.push_back(output);
#else
        std::fprintf(output_file, "%" PRIi64 " %" PRIi32 " %" PRIi32 " %" PRIi32 "\n", output.seed, output.x, output.z, output.score);
        std::fflush(output_file);
#endif
    };

    GpuOutputs gpu_outputs;
    CpuOutputs cpu_outputs;

#ifdef BOINC
    for (const auto &candidate : progress.pending) {
        gpu_outputs.queue.push(candidate);
    }
    progress.pending.clear();
#endif

#ifndef NO_GPU
    uint64_t start_seed = args.start_seed.value_or(random_start_seed());
    std::optional<uint64_t> end_seed;
    if (args.end_seed) {
        end_seed = (uint64_t)args.end_seed.value();
    }
    SeedIterator seed_range(start_seed, end_seed);
#ifdef BOINC
    seed_range.pos.store(progress.pos);
#endif

    std::vector<std::unique_ptr<GpuThread>> gpu_threads;
    if (!seed_range.exhausted()) {
        for (int device : args.devices) {
            gpu_threads.emplace_back(std::make_unique<GpuThread>(device, std::ref(seed_range), std::ref(gpu_outputs)));
        }
    }
    if (!gpu_threads.empty()) {
        std::printf("Starting from %" PRIi64 "\n", (int64_t)seed_range.pos.load());
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

#ifdef BOINC
    // Capture a consistent picture of the work: everything below the GPU's
    // done position has either produced a result (moved into progress.results)
    // or is still a pending candidate (queued or being tested by a CPU thread).
    // Lock order matches CpuThread: gpu_outputs, then cpu_outputs.
    auto snapshot = [&]() {
        std::lock_guard input_lock(gpu_outputs.mutex);
        progress.pending.clear();
        for (auto queue = gpu_outputs.queue; !queue.empty(); queue.pop()) {
            progress.pending.push_back(queue.front());
        }
        for (auto &thread : cpu_threads) {
            if (thread->current) progress.pending.push_back(*thread->current);
        }
        if (!gpu_threads.empty()) {
            progress.pos = gpu_threads[0]->done_pos.load(std::memory_order_relaxed);
        }
        std::lock_guard output_lock(cpu_outputs.mutex);
        while (!cpu_outputs.queue.empty()) {
            handle_result(cpu_outputs.queue.front());
            cpu_outputs.queue.pop();
        }
    };

    auto report_progress = [&]() {
        uint64_t done = gpu_threads.empty() ? progress.pos : gpu_threads[0]->done_pos.load(std::memory_order_relaxed);
        double fraction = (double)(done - progress.start) / (double)(progress.end - progress.start);
        // 100% is only reported once the output file exists.
        boinc_glue::fraction_done(std::min(fraction, 0.999));
    };
#endif

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    for (size_t i = 0; running.load(std::memory_order_relaxed); i++) {
        if (threads != 0) {
            std::lock_guard lock(cpu_outputs.mutex);
            while (!cpu_outputs.queue.empty()) {
                handle_result(cpu_outputs.queue.front());
                cpu_outputs.queue.pop();
            }
        }

        if (args.devices.size() == 0 && i % 10 == 0) {
            std::lock_guard lock(gpu_outputs.mutex);
            std::printf("gpu_outputs.queue.size() = %zu\n", gpu_outputs.queue.size());
        }

#ifndef NO_GPU
        if (seed_range.exhausted()) {
            running.store(false, std::memory_order_relaxed);
        }
#endif

#ifdef BOINC
        report_progress();
        if (boinc_glue::time_to_checkpoint()) {
            snapshot();
            if (boinc_glue::save_checkpoint(progress)) {
                boinc_glue::checkpoint_completed();
            }
        }
#endif

        if (!running.load(std::memory_order_relaxed)) break;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    const bool interrupted = signal_count.load(std::memory_order_relaxed) > 0;

    size_t pending_cpu_inputs = 0;
    {
        std::lock_guard lock(gpu_outputs.mutex);
        pending_cpu_inputs = gpu_outputs.queue.size();
    }
    size_t pending_cpu_outputs = 0;
    {
        std::lock_guard lock(cpu_outputs.mutex);
        pending_cpu_outputs = cpu_outputs.queue.size();
    }

    std::printf("\nShutting down...\n");
    if (pending_cpu_inputs > 0 || pending_cpu_outputs > 0) {
        std::printf("Pending CPU input queue: %zu, pending CPU output queue: %zu\n", pending_cpu_inputs, pending_cpu_outputs);
    }
    if (interrupted) {
        std::printf("Press Ctrl+C again to force exit immediately.\n");
    }

#ifndef NO_GPU
    for (auto &thread : gpu_threads) {
        (*thread).stop();
    }
#endif
#ifndef NO_NET
    if (client_thread) {
        (*client_thread).stop();
    }
    if (server_thread) {
        (*server_thread).shutdown();
    }
#endif

    // The GPU threads must be joined before the CPU threads are told to stop:
    // a CPU thread exits as soon as it sees an empty queue, so stopping it while
    // a GPU batch is still in flight would drop that batch's candidates.
#ifndef NO_GPU
    if (!gpu_threads.empty()) {
        std::printf("Waiting for GPU batches to finish...\n");
        for (auto &thread : gpu_threads) {
            (*thread).join();
        }
        {
            uint64_t total_seeds_checked = seed_range.pos.load(std::memory_order_relaxed) - start_seed;
            if (end_seed) total_seeds_checked = std::min(total_seeds_checked, *end_seed - start_seed);
            std::printf("Start seed: %" PRIi64 ", Total seeds checked: %" PRIu64 "\n", start_seed, total_seeds_checked);
        }
    }
#endif

#ifdef BOINC
    if (interrupted) {
        // Not finished: save where we are and leave without producing output.
        // The pending candidates are in the checkpoint, so drop them from the
        // queue and let the CPU threads exit after their current item.
        snapshot();
        boinc_glue::save_checkpoint(progress);
        {
            std::lock_guard lock(gpu_outputs.mutex);
            gpu_outputs.queue = {};
        }
        for (auto &thread : cpu_threads) (*thread).stop();
        for (auto &thread : cpu_threads) (*thread).join();
        std::fprintf(stderr, "Interrupted, progress saved to checkpoint\n");
        return 130;
    }
#endif

#ifndef NO_CPU
    for (auto &thread : cpu_threads) {
        (*thread).stop();
    }
    for (auto &thread : cpu_threads) {
        (*thread).join();
    }
    {
        std::lock_guard lock(cpu_outputs.mutex);
        while (!cpu_outputs.queue.empty()) {
            handle_result(cpu_outputs.queue.front());
            cpu_outputs.queue.pop();
        }
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

#ifdef BOINC
    progress.pending.clear();
    if (!gpu_threads.empty()) {
        progress.pos = gpu_threads[0]->done_pos.load(std::memory_order_relaxed);
    }
    if (!boinc_glue::write_output(output_path, progress)) {
        return 1;
    }
    boinc_glue::fraction_done(1.0);
    std::fprintf(stderr, "Done: %zu results\n", progress.results.size());
#else
    if (output_file != nullptr) {
        std::fclose(output_file);
    }
#endif

    return 0;
}

int main(int argc, char **argv) {
    int status;
    try {
        status = main_inner(argc, argv);
    } catch (std::exception &e) {
        std::fprintf(stderr, "Uncaught exception in main: %s\n", e.what());
        std::abort();
    }
#ifdef BOINC
    // Tells the client the task is over (status 0 = success); never returns.
    boinc_glue::finish(status);
#endif
    return status;
}
