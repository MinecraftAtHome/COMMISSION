#pragma once

// Thin wrapper around the BOINC API.
//
// main.cpp talks to BOINC only through this header so it never has to include
// the BOINC headers themselves (on Windows they pull in <windows.h>, which
// clashes with asio's winsock includes). Everything here compiles to nothing
// unless the build defines BOINC (make BOINC=1).

#ifdef BOINC

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

namespace boinc_glue {

// Must be the first thing main() does, before any threads are started.
void init();

// GPU assigned by the BOINC client, or -1 when running standalone / unknown.
int assigned_gpu_device();

bool is_standalone();

// Map a logical file name from the workunit/result template to its real path.
std::string resolve_filename(const std::string &logical_name);

void fraction_done(double fraction);
bool time_to_checkpoint();
void checkpoint_completed();

// Report success and exit. Does not return.
[[noreturn]] void finish(int status);
// Report an error with a message shown to the user and exit. Does not return.
[[noreturn]] void finish_message(int status, const char *message);
// Ask the client to restart the task later (e.g. GPU temporarily unusable).
[[noreturn]] void temporary_exit(int delay_seconds, const char *reason);

// Everything needed to resume a workunit where it left off.
struct Checkpoint {
    // Identity of the workunit; a checkpoint for different parameters is ignored.
    uint64_t start = 0;
    uint64_t end = 0;
    int32_t min_size = 0;
    bool large_biomes = false;
    bool unbound = false;

    // All seeds in [start, pos) have been through the GPU filter.
    uint64_t pos = 0;
    // Final results found so far.
    std::vector<CpuOutput> results;
    // GPU candidates from [start, pos) not yet checked by the CPU.
    std::vector<GpuOutput> pending;

    bool same_workunit(const Checkpoint &other) const {
        return start == other.start && end == other.end && min_size == other.min_size
            && large_biomes == other.large_biomes && unbound == other.unbound;
    }
};

// Load the checkpoint in the slot directory. Returns false if there is none or
// it is unreadable; `out` is only modified on success.
bool load_checkpoint(Checkpoint &out);
// Atomically replace the checkpoint (write temp file, then rename), inside a
// BOINC critical section so the client cannot kill us half way through.
bool save_checkpoint(const Checkpoint &checkpoint);

// Write the final result file. Results are sorted so two hosts that crunched the
// same workunit produce byte-identical files (needed for bitwise validation).
bool write_output(const std::string &path, const Checkpoint &checkpoint);

} // namespace boinc_glue

#endif
