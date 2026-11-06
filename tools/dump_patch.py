#!/usr/bin/env python3
"""Insert a candidate-set dump into gpu.cu (identical text for any tree). With
COMMISSION_DUMP=<file> and COMMISSION_DUMP_ITERS=<n>, after each of 5 mid-funnel stages the
stream is synced and "<stage> <seed> <x> <z>" lines are written (before later stages reuse the
buffer); the process exits after n iterations. Used to prove set identity against upstream."""
import sys
p = sys.argv[1]; s = open(p).read()
assert "COMMISSION_DUMP" not in s
def ins_after(anchor, text, count=1):
    global s
    assert s.count(anchor) >= 1, anchor
    s = s.replace(anchor, anchor + text, count)
ins_after("  auto start = std::chrono::steady_clock::now();\n", r'''
  std::FILE *dump_file = std::getenv("COMMISSION_DUMP") ? std::fopen(std::getenv("COMMISSION_DUMP"), "w") : nullptr;
  auto dump_stage = [&](const char *tag, const SeedPos *data, size_t len_offset) {
    if (!dump_file) return;
    TRY_CUDA(::cudaStreamSynchronize(stream));
    BufferLens lens;
    TRY_CUDA(cudaMemcpy(&lens, device_buffer_lens, sizeof(lens), cudaMemcpyDeviceToHost));
    const uint32_t nseeds = lens.results_len_filter_seeds;
    const uint32_t len = *reinterpret_cast<const uint32_t *>(reinterpret_cast<const char *>(&lens) + len_offset);
    std::vector<uint64_t> seeds(nseeds);
    TRY_CUDA(cudaMemcpy(seeds.data(), outputs_filter_seeds.data, nseeds * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    std::vector<SeedPos> v(len);
    TRY_CUDA(cudaMemcpy(v.data(), data, len * sizeof(SeedPos), cudaMemcpyDeviceToHost));
    for (const auto &e : v) std::fprintf(dump_file, "%s %llu %d %d\n", tag, (unsigned long long)seeds[e.seed_index], e.x, e.z);
  };
  uint32_t dumped = 0;
''')
ins_after("    stage_filter_2_0a.record(stream);\n", '    dump_stage("2_01a", outputs_filter_2_0a.data, offsetof(BufferLens, results_len_filter_2_0a));\n')
ins_after("    stage_filter_gradvecs_2.record(stream);\n", '    dump_stage("gv2", outputs_filter_gradvecs_2.data, offsetof(BufferLens, results_len_filter_gradvecs_2));\n')
ins_after("    stage_filter_2_0b.record(stream);\n", '    dump_stage("2_01b", outputs_filter_2_0b.data, offsetof(BufferLens, results_len_filter_2_0b));\n')
ins_after("        filter.stage.record(stream);\n", '        if (filter_index < 2) dump_stage(filter_index == 0 ? "2a" : "2b", filter.outputs.data, offsetof(BufferLens, results_len_filter_2) + filter_index * sizeof(uint32_t));\n')
ins_after("    TRY_CUDA(cudaStreamSynchronize(stream));\n", '    if (dump_file && ++dumped == (uint32_t)std::atoi(std::getenv("COMMISSION_DUMP_ITERS"))) { std::fclose(dump_file); std::_Exit(0); }\n')
for inc in ("#include <cstdlib>\n", "#include <cstddef>\n", "#include <vector>\n"):
    if inc not in s: s = s.replace("#include <cstdio>\n", "#include <cstdio>\n" + inc, 1)
open(p, "w").write(s)
