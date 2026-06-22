#include "cpu.h"
#include "cubiomes.h"

#include <cinttypes>
#include <optional>
#include <chrono>

std::optional<CpuOutput> process(Cubiomes *cubiomes, int32_t min_size, const GpuOutput &input) {
    // return {{ input.seed, input.x, input.z, 0 }};

    cubiomes_apply_seed(cubiomes, input.seed);

    int32_t range = 12800 * (large_biomes ? 4 : 1);

    if (!cubiomes_test_monte_carlo(cubiomes, input.x, input.z, range, (min_size * 0.9), 0.999)) {
        // std::printf("Test %" PRIi64 " %" PRIi32 " %" PRIi32 " failed monteCarloBiomes\n", input.seed, input.x, input.z);
        return {};
    }

    if (!cubiomes_test_biome_centers(cubiomes, input.x, input.z, range, min_size, 16, 4, nullptr)) {
        // std::printf("Test %" PRIi64 " %" PRIi32 " %" PRIi32 " failed getBiomeCenters at scale 16\n", input.seed, input.x, input.z);
        return {};
    }

    PosArea res;
    if (!cubiomes_test_biome_centers(cubiomes, input.x, input.z, range, min_size, 4, 2, &res)) {
        // std::printf("Test %" PRIi64 " %" PRIi32 " %" PRIi32 " failed getBiomeCenters at scale 4\n", input.seed, input.x, input.z);
        return {};
    }

    // std::printf("Test %" PRIi64 " %" PRIi32 " %" PRIi32 " passed\n", input.seed, input.x, input.z);
    return {{ input.seed, res.x, res.z, res.area }};
}

CpuThread::CpuThread(int id, int32_t min_size, GpuOutputs &inputs, CpuOutputs &outputs) : Thread(), id(id), min_size(min_size), inputs(inputs), outputs(outputs) {
    start();
}

void CpuThread::run() {
    std::printf("Started cpu thread %d\n", id);

    Cubiomes *cubiomes = cubiomes_create(large_biomes);

    while (!should_stop()) {
        GpuOutput input;
        {
            std::unique_lock lock(inputs.mutex);
            if (inputs.queue.empty()) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            input = inputs.queue.front();
            inputs.queue.pop();
        }

        const auto start = std::chrono::steady_clock::now();

        const auto output = process(cubiomes, min_size, input);

        const auto end = std::chrono::steady_clock::now();
        double time_total = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;
        // std::printf("Cpu test took %.3f s\n", time_total);

        if (!output) continue;

        {
            std::lock_guard lock(outputs.mutex);
            outputs.queue.push(output.value());
        }
    }

    cubiomes_free(cubiomes);
}
