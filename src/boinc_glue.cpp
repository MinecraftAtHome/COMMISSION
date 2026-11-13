#ifdef BOINC

#include "boinc_glue.h"

#ifdef _WIN32
#include "boinc_win.h"
#endif
#include "boinc_api.h"
#include "filesys.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>

namespace boinc_glue {

namespace {

constexpr const char *checkpoint_path = "commission_checkpoint.txt";
constexpr const char *checkpoint_tmp_path = "commission_checkpoint.txt.tmp";
constexpr const char *checkpoint_magic = "COMMISSION-CHECKPOINT";
constexpr int checkpoint_version = 1;

APP_INIT_DATA init_data;

} // namespace

void init() {
    BOINC_OPTIONS options;
    boinc_options_defaults(options);
    // The CPU threads that verify GPU candidates must not be starved by the
    // client's idle-priority default, or the GPU would sit waiting on them.
    options.normal_thread_priority = true;
    int status = boinc_init_options(&options);
    if (status != 0) {
        std::fprintf(stderr, "boinc_init_options failed: %d\n", status);
        std::exit(status);
    }
    boinc_get_init_data(init_data);
}

int assigned_gpu_device() {
    return init_data.gpu_device_num;
}

bool is_standalone() {
    return boinc_is_standalone() != 0;
}

std::string resolve_filename(const std::string &logical_name) {
    std::string path;
    if (boinc_resolve_filename_s(logical_name.c_str(), path) != 0) {
        // In standalone mode there is no soft link file; use the name as-is.
        return logical_name;
    }
    return path;
}

void fraction_done(double fraction) {
    boinc_fraction_done(std::clamp(fraction, 0.0, 1.0));
}

bool time_to_checkpoint() {
    return boinc_time_to_checkpoint() != 0;
}

void checkpoint_completed() {
    boinc_checkpoint_completed();
}

void finish(int status) {
    boinc_finish(status);
    std::exit(status); // boinc_finish never returns; this keeps [[noreturn]] honest.
}

void finish_message(int status, const char *message) {
    boinc_finish_message(status, message, false);
    std::exit(status);
}

void temporary_exit(int delay_seconds, const char *reason) {
    boinc_temporary_exit(delay_seconds, reason, false);
    std::exit(0);
}

bool load_checkpoint(Checkpoint &out) {
    std::FILE *file = boinc_fopen(checkpoint_path, "r");
    if (file == nullptr) return false;

    Checkpoint checkpoint;
    bool ok = false;
    do {
        char magic[32] = {};
        int version = 0;
        if (std::fscanf(file, "%31s %d", magic, &version) != 2) break;
        if (std::strcmp(magic, checkpoint_magic) != 0 || version != checkpoint_version) break;

        int large_biomes = 0, unbound = 0;
        if (std::fscanf(file, " config %" SCNu64 " %" SCNu64 " %" SCNi32 " %d %d",
                        &checkpoint.start, &checkpoint.end, &checkpoint.min_size, &large_biomes, &unbound) != 5) break;
        checkpoint.large_biomes = large_biomes != 0;
        checkpoint.unbound = unbound != 0;

        if (std::fscanf(file, " pos %" SCNu64, &checkpoint.pos) != 1) break;

        size_t count = 0;
        if (std::fscanf(file, " results %zu", &count) != 1) break;
        checkpoint.results.resize(count);
        size_t read = 0;
        for (auto &r : checkpoint.results) {
            if (std::fscanf(file, " %" SCNu64 " %" SCNi32 " %" SCNi32 " %" SCNi32, &r.seed, &r.x, &r.z, &r.score) != 4) break;
            read++;
        }
        if (read != count) break;

        if (std::fscanf(file, " pending %zu", &count) != 1) break;
        checkpoint.pending.resize(count);
        read = 0;
        for (auto &p : checkpoint.pending) {
            if (std::fscanf(file, " %" SCNu64 " %" SCNi32 " %" SCNi32, &p.seed, &p.x, &p.z) != 3) break;
            read++;
        }
        if (read != count) break;

        // The trailer guards against a truncated file.
        char trailer[8] = {};
        if (std::fscanf(file, " %7s", trailer) != 1 || std::strcmp(trailer, "end") != 0) break;

        ok = true;
    } while (false);
    std::fclose(file);

    if (!ok) {
        std::fprintf(stderr, "Ignoring unreadable checkpoint %s\n", checkpoint_path);
        return false;
    }
    out = std::move(checkpoint);
    return true;
}

bool save_checkpoint(const Checkpoint &checkpoint) {
    boinc_begin_critical_section();

    bool ok = false;
    std::FILE *file = boinc_fopen(checkpoint_tmp_path, "w");
    if (file != nullptr) {
        std::fprintf(file, "%s %d\n", checkpoint_magic, checkpoint_version);
        std::fprintf(file, "config %" PRIu64 " %" PRIu64 " %" PRIi32 " %d %d\n",
                     checkpoint.start, checkpoint.end, checkpoint.min_size,
                     checkpoint.large_biomes ? 1 : 0, checkpoint.unbound ? 1 : 0);
        std::fprintf(file, "pos %" PRIu64 "\n", checkpoint.pos);
        std::fprintf(file, "results %zu\n", checkpoint.results.size());
        for (const auto &r : checkpoint.results) {
            std::fprintf(file, "%" PRIu64 " %" PRIi32 " %" PRIi32 " %" PRIi32 "\n", r.seed, r.x, r.z, r.score);
        }
        std::fprintf(file, "pending %zu\n", checkpoint.pending.size());
        for (const auto &p : checkpoint.pending) {
            std::fprintf(file, "%" PRIu64 " %" PRIi32 " %" PRIi32 "\n", p.seed, p.x, p.z);
        }
        std::fprintf(file, "end\n");
        ok = std::fflush(file) == 0 && !std::ferror(file);
        ok = (std::fclose(file) == 0) && ok;
        ok = ok && boinc_rename(checkpoint_tmp_path, checkpoint_path) == 0;
    }
    if (!ok) {
        std::fprintf(stderr, "Failed to write checkpoint %s\n", checkpoint_path);
    }

    boinc_end_critical_section();
    return ok;
}

bool write_output(const std::string &path, const Checkpoint &checkpoint) {
    std::vector<CpuOutput> results = checkpoint.results;
    auto key = [](const CpuOutput &r) { return std::make_tuple(r.seed, r.x, r.z, r.score); };
    std::sort(results.begin(), results.end(), [&](const CpuOutput &a, const CpuOutput &b) { return key(a) < key(b); });
    // Neighbouring GPU candidates of one seed can converge on the same island.
    results.erase(std::unique(results.begin(), results.end(), [&](const CpuOutput &a, const CpuOutput &b) { return key(a) == key(b); }), results.end());

    boinc_begin_critical_section();
    bool ok = false;
    // Binary mode: Windows text mode would write CRLF, and the file must be
    // byte-identical across platforms for bitwise validation.
    std::FILE *file = boinc_fopen(path.c_str(), "wb");
    if (file != nullptr) {
        // Seeds are printed signed, matching the standalone output.txt format
        // and the seed you type into Minecraft.
        for (const auto &r : results) {
            std::fprintf(file, "%" PRIi64 " %" PRIi32 " %" PRIi32 " %" PRIi32 "\n", (int64_t)r.seed, r.x, r.z, r.score);
        }
        // Trailer: proves the task ran to completion (an empty file would be
        // indistinguishable from a crash) and identifies the range covered.
        std::fprintf(file, "done %" PRIi64 " %" PRIi64 " %" PRIi32 " %s %s\n",
                     (int64_t)checkpoint.start, (int64_t)checkpoint.end, checkpoint.min_size,
                     checkpoint.large_biomes ? "lb" : "sb", checkpoint.unbound ? "unbound" : "bounded");
        ok = std::fflush(file) == 0 && !std::ferror(file);
        ok = (std::fclose(file) == 0) && ok;
    }
    boinc_end_critical_section();

    if (!ok) {
        std::fprintf(stderr, "Failed to write output file %s\n", path.c_str());
    }
    return ok;
}

} // namespace boinc_glue

#endif
