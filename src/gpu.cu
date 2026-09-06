#include "Random.h"
#include "gpu.h"

#include <array>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>
#include <utility>
#include <algorithm>

#define PANIC(...)                                                             \
  {                                                                            \
    std::fprintf(stderr, __VA_ARGS__);                                         \
    std::abort();                                                              \
  }

#define TRY_CUDA(expr) try_cuda(expr, __FILE__, __LINE__)

void try_cuda(cudaError_t error, const char *file, uint64_t line) {
  if (error == cudaSuccess)
    return;

  PANIC("CUDA error at %s:%" PRIu64 ": %s (%d)\n", file, line,
        cudaGetErrorString(error), error);
}

// from cubiomes
constexpr XrsrForkHash hash_continentalness{
    0x83886c9d0ae3a662, 0xafa638a61b42e8ad}; // md5 "minecraft:continentalness"
constexpr XrsrForkHash hash_continentalness_large{
    0x9a3f51a113fce8dc,
    0xee2dbd157e5dcdad}; // md5 "minecraft:continentalness_large"
constexpr XrsrForkHash hash_octave[]{
    {0xb198de63a8012672, 0x7b84cad43ef7b5a8}, // md5 "octave_-12"
    {0x0fd787bfbc403ec3, 0x74a4a31ca21b48b8}, // md5 "octave_-11"
    {0x36d326eed40efeb2, 0x5be9ce18223c636a}, // md5 "octave_-10"
    {0x082fe255f8be6631, 0x4e96119e22dedc81}, // md5 "octave_-9"
    {0x0ef68ec68504005e, 0x48b6bf93a2789640}, // md5 "octave_-8"
    {0xf11268128982754f, 0x257a1d670430b0aa}, // md5 "octave_-7"
    {0xe51c98ce7d1de664, 0x5f9478a733040c45}, // md5 "octave_-6"
    {0x6d7b49e7e429850a, 0x2e3063c622a24777}, // md5 "octave_-5"
    {0xbd90d5377ba1b762, 0xc07317d419a7548d}, // md5 "octave_-4"
    {0x53d39c6752dac858, 0xbcd1c5a80ab65b3e}, // md5 "octave_-3"
    {0xb4a24d7a84e7677b, 0x023ff9668e89b5c4}, // md5 "octave_-2"
    {0xdffa22b534c5f608, 0xb9b67517d3665ca9}, // md5 "octave_-1"
    {0xd50708086cef4d7c, 0x6e1651ecc7f43309}, // md5 "octave_0"
};

struct alignas(16) ImprovedNoise {
  uint8_t p[256];
  float xo;
  float yo;
  float zo;
  float pad;
};

struct Octave {
  ImprovedNoise noise;
  double input_factor;
  double value_factor;
};

template <size_t N> struct NoiseParameters {
  int32_t first_octave;
  std::array<double, N> amplitudes;
};

template <size_t N>
constexpr NoiseParameters<N>
make_noise_parameters(int32_t first_octave, const double (&amplitudes)[N]) {
  std::array<double, N> amp{};
  std::copy(std::begin(amplitudes), std::end(amplitudes), amp.begin());
  return {first_octave, amp};
}

constexpr auto continentalness_parameters = make_noise_parameters(-9, {1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0});
constexpr auto continentalness_large_parameters = make_noise_parameters(-11, {1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0});

struct OctaveConfig {
  XrsrForkHash fork_hash;
  double input_factor;
  double value_factor;
};

template <size_t N> struct NormalNoiseConfig {
  XrsrForkHash fork_hash;
  std::array<OctaveConfig, N> octaves_a;
  std::array<OctaveConfig, N> octaves_b;
};

template <size_t N>
constexpr NormalNoiseConfig<N>
make_normal_noise_config(const NoiseParameters<N> &noise_parameters, const XrsrForkHash &fork_hash) {
  NormalNoiseConfig<N> res{fork_hash};

  const auto first_octave = noise_parameters.first_octave;
  const auto &amplitudes = noise_parameters.amplitudes;

  double root_value_factor = 0.16666666666666666 / (0.1 * (1.0 + 1.0 / amplitudes.size()));

  double input_factor = 1.0 / (1 << -first_octave);
  double value_factor = (1 << (amplitudes.size() - 1)) / ((1 << amplitudes.size()) - 1.0) * root_value_factor;

  for (size_t i = 0; i < amplitudes.size(); i++) {
    res.octaves_a[i] = {hash_octave[first_octave + 12 + i], input_factor, value_factor * amplitudes[i]};
    res.octaves_b[i] = {hash_octave[first_octave + 12 + i], input_factor * 1.0181268882175227, value_factor * amplitudes[i]};
    input_factor *= 2.0;
    value_factor *= 0.5;
  }

  return res;
}

__device__ constexpr auto continentalness_config = make_normal_noise_config(continentalness_parameters, hash_continentalness);
__device__ constexpr auto continentalness_large_config = make_normal_noise_config(continentalness_large_parameters, hash_continentalness_large);
__device__ constexpr auto chosen_continentalness_config = large_biomes ? continentalness_large_config : continentalness_config;
__device__ constexpr auto device_chosen_continentalness_config = chosen_continentalness_config;

struct GradDotTable {
  float x[16];
  float y[16];
  float z[16];
};

__device__ GradDotTable device_grad_dot_table;

void init_grad_dot_table() {
  GradDotTable table;
  table.x[0] = 1;
  table.y[0] = 1;
  table.z[0] = 0; // { 1,  1,  0}
  table.x[1] = -1;
  table.y[1] = 1;
  table.z[1] = 0; // {-1,  1,  0}
  table.x[2] = 1;
  table.y[2] = -1;
  table.z[2] = 0; // { 1, -1,  0}
  table.x[3] = -1;
  table.y[3] = -1;
  table.z[3] = 0; // {-1, -1,  0}
  table.x[4] = 1;
  table.y[4] = 0;
  table.z[4] = 1; // { 1,  0,  1}
  table.x[5] = -1;
  table.y[5] = 0;
  table.z[5] = 1; // {-1,  0,  1}
  table.x[6] = 1;
  table.y[6] = 0;
  table.z[6] = -1; // { 1,  0, -1}
  table.x[7] = -1;
  table.y[7] = 0;
  table.z[7] = -1; // {-1,  0, -1}
  table.x[8] = 0;
  table.y[8] = 1;
  table.z[8] = 1; // { 0,  1,  1}
  table.x[9] = 0;
  table.y[9] = -1;
  table.z[9] = 1; // { 0, -1,  1}
  table.x[10] = 0;
  table.y[10] = 1;
  table.z[10] = -1; // { 0,  1, -1}
  table.x[11] = 0;
  table.y[11] = -1;
  table.z[11] = -1; // { 0, -1, -1}
  table.x[12] = 1;
  table.y[12] = 1;
  table.z[12] = 0; // { 1,  1,  0}
  table.x[13] = 0;
  table.y[13] = -1;
  table.z[13] = 1; // { 0, -1,  1}
  table.x[14] = -1;
  table.y[14] = 1;
  table.z[14] = 0; // {-1,  1,  0}
  table.x[15] = 0;
  table.y[15] = -1;
  table.z[15] = -1; // { 0, -1, -1}

  void *device_grad_dot_table_addr;
  TRY_CUDA(cudaGetSymbolAddress(&device_grad_dot_table_addr, device_grad_dot_table));
  TRY_CUDA(cudaMemcpy(device_grad_dot_table_addr, &table, sizeof(GradDotTable), cudaMemcpyHostToDevice));
}

__forceinline__ __device__ float gradDot(const GradDotTable &table, uint8_t p, float x, float y, float z) {
  const uint32_t hash = p & 0xF;
  return fmaf(x, table.x[hash], fmaf(y, table.y[hash], z * table.z[hash]));
}

__forceinline__ __device__ float smoothstep(float value) {
  return value * value * value * (value * (value * 6.0f - 15.0f) + 10.0f);
}

__forceinline__ __device__ float lerp1(float fx, float v0, float v1) {
  return fmaf(fx, v1 - v0, v0);
}

__forceinline__ __device__ float lerp2(float fx, float fy, float v00, float v10, float v01, float v11) {
  return lerp1(fy, lerp1(fx, v00, v10), lerp1(fx, v01, v11));
}

__forceinline__ __device__ float lerp3(float fx, float fy, float fz, float v000, float v100, float v010, float v110, float v001, float v101, float v011, float v111) {
  return lerp1(fz, lerp2(fx, fy, v000, v100, v010, v110), lerp2(fx, fy, v001, v101, v011, v111));
}

__device__ float sample_noise(const GradDotTable &table, const ImprovedNoise &noise, float x, float y, float z) {
  x += noise.xo;
  y += noise.yo;
  z += noise.zo;
  int32_t int_x = __float2int_rd(x);
  int32_t int_y = __float2int_rd(y);
  int32_t int_z = __float2int_rd(z);
  float frac_x = x - (float)int_x;
  float frac_y = y - (float)int_y;
  float frac_z = z - (float)int_z;
  uint8_t p0 = noise.p[(int_x) & 0xFF];
  uint8_t p1 = noise.p[(int_x + 1) & 0xFF];
  uint8_t p00 = noise.p[(p0 + int_y) & 0xFF];
  uint8_t p01 = noise.p[(p0 + int_y + 1) & 0xFF];
  uint8_t p10 = noise.p[(p1 + int_y) & 0xFF];
  uint8_t p11 = noise.p[(p1 + int_y + 1) & 0xFF];
  float n000 = gradDot(table, noise.p[(p00 + int_z) & 0xFF], frac_x, frac_y, frac_z);
  float n100 = gradDot(table, noise.p[(p10 + int_z) & 0xFF], frac_x - 1.0f, frac_y, frac_z);
  float n010 = gradDot(table, noise.p[(p01 + int_z) & 0xFF], frac_x, frac_y - 1.0f, frac_z);
  float n110 = gradDot(table, noise.p[(p11 + int_z) & 0xFF], frac_x - 1.0f, frac_y - 1.0f, frac_z);
  float n001 = gradDot(table, noise.p[(p00 + int_z + 1) & 0xFF], frac_x, frac_y, frac_z - 1.0f);
  float n101 = gradDot(table, noise.p[(p10 + int_z + 1) & 0xFF], frac_x - 1.0f, frac_y, frac_z - 1.0f);
  float n011 = gradDot(table, noise.p[(p01 + int_z + 1) & 0xFF], frac_x, frac_y - 1.0f, frac_z - 1.0f);
  float n111 = gradDot(table, noise.p[(p11 + int_z + 1) & 0xFF], frac_x - 1.0f, frac_y - 1.0f, frac_z - 1.0f);
  float fx = smoothstep(frac_x);
  float fy = smoothstep(frac_y);
  float fz = smoothstep(frac_z);
  return lerp3(fx, fy, fz, n000, n100, n010, n110, n001, n101, n011, n111);
}

__device__ float wrap(float value) {
  // return value - std::floor(value / 256.0) * 256.0;
  return value;
}

template <OctaveConfig config>
__forceinline__ __device__ float sample_octave(const GradDotTable &table, const ImprovedNoise &noise, int32_t x, int32_t y, int32_t z) {
  return sample_noise(table, noise, wrap(x * (float)config.input_factor), wrap(y * (float)config.input_factor), wrap(z * (float)config.input_factor)) * (float)config.value_factor;
}

__device__ void init_noise(ImprovedNoise &noise, XrsrRandom &&random) {
  noise.xo = random.nextFloat() * 256.0f;
  noise.yo = random.nextFloat() * 256.0f;
  noise.zo = random.nextFloat() * 256.0f;

  for (uint32_t i = 0; i < 256; i++) {
    noise.p[i] = i;
  }
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t j = random.nextInt(256 - i);
    uint8_t b = noise.p[i];
    noise.p[i] = noise.p[i + j];
    noise.p[i + j] = b;
  }
}

struct DeviceBuffer {
  void *data;
  size_t size;

  DeviceBuffer(size_t size) : size(size) { TRY_CUDA(cudaMalloc(&data, size)); }

  ~DeviceBuffer() { TRY_CUDA(cudaFree(data)); }
};

template <typename T> struct OutputBuffer {
  T *data;
  uint32_t *len;
  uint32_t max_len;

  OutputBuffer(T *data, uint32_t *len, uint32_t max_len)
      : data(data), len(len), max_len(max_len) {}

  OutputBuffer(const DeviceBuffer &buffer, uint32_t *len)
      : data((T *)buffer.data), len(len), max_len(buffer.size / sizeof(T)) {}

  OutputBuffer(const OutputBuffer<T> &other)
      : data(other.data), len(other.len), max_len(other.max_len) {}
};

template <typename T> struct InputBuffer {
  const T *data;
  const uint32_t *len;

  InputBuffer(const T *data, const uint32_t *len) : data(data), len(len) {}

  InputBuffer(const OutputBuffer<T> &buffer)
      : data(buffer.data), len(buffer.len) {}

  InputBuffer(const InputBuffer<T> &other) : data(other.data), len(other.len) {}
};

__device__ inline uint64_t mix64_finish(uint64_t x) {
  x = (x ^ (x >> 27)) * XrsrRandom::XRSR_MIX2;
  return x ^ (x >> 31);
}

// fork() runs two full state updates, but the second is only ever needed by the
// B fork -- the two values it returns are already fixed before it happens.
// filter_seeds defers the B fork to its drain, so in the prologue that update is
// pure waste on the 78% of seeds that never get there. fork_short stops one
// update short; advance_one supplies it in the drain.
__device__ inline XrsrRandomFork fork_short(XrsrRandom &rng) {
  const uint64_t l = rng.lo, h = rng.hi;
  const uint64_t r1 = XrsrRandom::rol64(l + h, 17) + l;
  const uint64_t hx = h ^ l;
  const uint64_t l2 = XrsrRandom::rol64(l, 49) ^ hx ^ (hx << 21);
  const uint64_t h2 = XrsrRandom::rol64(hx, 28);
  const uint64_t r2 = XrsrRandom::rol64(l2 + h2, 17) + l2;
  rng.lo = l2;
  rng.hi = h2;
  return { r1, r2 };
}

__device__ inline void advance_one(XrsrRandom &rng) {
  const uint64_t l = rng.lo;
  const uint64_t hx = rng.hi ^ l;
  rng.lo = XrsrRandom::rol64(l, 49) ^ hx ^ (hx << 21);
  rng.hi = XrsrRandom::rol64(hx, 28);
}

__device__ inline XrsrRandomFork XrsrRandom_seed_fork_from_lh(uint64_t l, uint64_t h) {
  uint64_t r1 = XrsrRandom::rol64(l + h, 17) + l;
  h ^= l;
  uint64_t l2 = XrsrRandom::rol64(l, 49) ^ h ^ (h << 21);
  uint64_t h2 = XrsrRandom::rol64(h, 28);
  uint64_t r2 = XrsrRandom::rol64(l2 + h2, 17) + l2;
  return { r1, r2 };
}

__device__ inline XrsrRandomFork XrsrRandom_seed_fork(uint64_t seed) {
  seed ^= XrsrRandom::XRSR_SILVER_RATIO;
  uint64_t l = XrsrRandom::mix64(seed);
  uint64_t h = XrsrRandom::mix64(seed + XrsrRandom::XRSR_GOLDEN_RATIO);
  
  uint64_t r1 = XrsrRandom::rol64(l + h, 17) + l;
  
  // update state once
  h ^= l;
  uint64_t l2 = XrsrRandom::rol64(l, 49) ^ h ^ (h << 21);
  uint64_t h2 = XrsrRandom::rol64(h, 28);
  
  // skip state update
  uint64_t r2 = XrsrRandom::rol64(l2 + h2, 17) + l2;

  return { r1, r2 };
}

__device__ inline void XrsrRandom_double_fork(XrsrRandom &rng, XrsrRandomFork &fork_a, XrsrRandomFork &fork_b) {
  uint64_t l = rng.lo;
  uint64_t h = rng.hi;
  
  // fork A
  uint64_t r1 = XrsrRandom::rol64(l + h, 17) + l;
  h ^= l;
  uint64_t l2 = XrsrRandom::rol64(l, 49) ^ h ^ (h << 21);
  uint64_t h2 = XrsrRandom::rol64(h, 28);
  
  uint64_t r2 = XrsrRandom::rol64(l2 + h2, 17) + l2;
  h2 ^= l2;
  uint64_t l3 = XrsrRandom::rol64(l2, 49) ^ h2 ^ (h2 << 21);
  uint64_t h3 = XrsrRandom::rol64(h2, 28);
  
  fork_a = { r1, r2 };
  
  // fork B, skip 4th state update
  uint64_t r3 = XrsrRandom::rol64(l3 + h3, 17) + l3;
  h3 ^= l3;
  uint64_t l4 = XrsrRandom::rol64(l3, 49) ^ h3 ^ (h3 << 21);
  uint64_t h4 = XrsrRandom::rol64(h3, 28);
  
  uint64_t r4 = XrsrRandom::rol64(l4 + h4, 17) + l4;
  
  fork_b = { r3, r4 };
}

namespace KernelFilterSeeds {
constexpr uint32_t threads_per_block = 256;
constexpr uint32_t threads_per_run = UINT64_C(1) << 28; //28
constexpr uint32_t seeds_per_thread = 32;

__device__ XrsrRandomFork noise_yo_fork(XrsrRandomFork noise_fork) {
  uint64_t l = noise_fork.lo;
  uint64_t h = noise_fork.hi;
  
  // skip r
  h ^= l;
  return {
      XrsrRandom::rol64(l, 49) ^ h ^ (h << 21),
      XrsrRandom::rol64(h, 28)
  };
}

constexpr XrsrForkHash octave_yo_fork_hash(XrsrForkHash hash) {
  XrsrRandom rng{hash.lo, hash.hi};
  rng.nextInternal();
  return {rng.lo, rng.hi};
}

template <OctaveConfig octave_config>
__device__ float octave_yo_mod1(const XrsrRandomFork &noise_yo_fork) {
  constexpr auto fork_hash = octave_yo_fork_hash(octave_config.fork_hash);

  // skip state update
  uint64_t l = noise_yo_fork.lo ^ fork_hash.lo;
  uint64_t h = noise_yo_fork.hi ^ fork_hash.hi;
  uint64_t r = XrsrRandom::rol64(l + h, 17) + l;
  
  return (uint32_t)((r >> 32) & 0xFFFFFF) * 5.9604645E-8f;
}

__global__ __launch_bounds__(threads_per_block) void kernel(uint64_t start_seed, OutputBuffer<uint64_t> outputs) {
  // 0.045f, 0.035f, 0.03f, 0.025f  ==  1 in 2700, 9400, 26000, 54000
  //
  // Swept 0.030-0.045 on a GTX TITAN X against a neutral start region
  // (6.87e10 seeds per run, PRINT_INTERVAL 256). 0.038 is the peak of the
  // discovery *rate* - candidates reaching filter_2a/2b/2c per wall second -
  // and every other value measured worse on it:
  //
  //   maxScore   time    scan/s   survivors    2b/s   rate vs 0.038
  //   0.045     20.26s   3.392G   244.14ppm   57.21      -23.3%
  //   0.038     12.86s   5.343G   159.19ppm   74.57       peak
  //   0.035     10.34s   6.644G   105.57ppm   63.15      -15.3%
  //   0.032      8.48s   8.106G    67.17ppm   52.24      -29.9%
  //   0.030      7.57s   9.080G    48.34ppm   44.39      -40.5%
  //
  // Tightening buys raw throughput (9.1 vs 5.3 Gseed/s at 0.030) but drops more
  // yield than it gains speed, so it is a net loss for a search that runs
  // continuously.
  //
  // Loosening is the profitable direction, and by more than it first appeared.
  // filter_seeds costs the same whatever this value is -- every seed is hashed
  // before any threshold can reject it -- so an extra survivor costs only the
  // downstream work. Both earlier sweeps were also measured against a survivor
  // ceiling that silently truncated the loose end.
  //
  // Reswept with the ceiling at 1<<19 and PRINT_INTERVAL 512, three rounds of a
  // round-robin, quoting discovery rate (finds per wall second) against 0.038:
  //
  //   maxScore  ms/iter  surv/iter  %cap    filter_2a       filter_2b       filter_2c
  //   0.038       30.52      42709   8.1%       -               -               -
  //   0.046       46.77     110497  21.1%  +43.0% (+-0.6)  +34.7% (+-2.3)  +24.5% (+-10.9)
  //   0.054       77.45     247008  47.1%  +56.4% (+-0.6)  +40.9% (+-2.1)  +19.2% (+-10.2)
  //   0.062      132.09     495352  94.5%  +45.6% (+-0.6)  +26.4% (+-2.0)   -6.5% (+- 9.8)
  //
  // There is a real peak: 0.062 is past it at every stage. 0.054 is the choice --
  // about 41% more finds a second than 0.038, for 2.5x the wall time an
  // iteration, and verify_seeds.py still reports the same 7 of 10 reference
  // seeds. ms/iteration gets worse; islands per hour, which is what the search
  // is for, gets substantially better.
  //
  // Note this inverts the optimisation target. At 0.038 filter_seeds was 64.7%
  // of the run and effectively at its floor; at 0.054 it is 27.6%, while
  // gradvecs_1 (34.5%) and filter_2_01a (24.4%) carry 59% between them.
  constexpr float maxScore = 0.054f;

  // Stack-A accumulator only: with the run 32-aligned, s = S0 ^ j, so the input
  // to mix64(s)'s first multiply is X0 ^ j = P1 + r for r = (X0 & 31) ^ j.
  // Walking r rather than j makes that an arithmetic progression, so its product
  // with MIX1 is an accumulator: one multiply for the run, one add per seed.
  //
  // The a0 test rejects 78% of seeds, but it rejects them per *lane*, and a warp
  // costs the same whether one lane is live or all 32. With a0 passing 21.7%,
  // 1 - (1-0.217)^32 = 99.97% of warps reach the second fork regardless, so that
  // fork -- and the whole tail behind it -- is paid at full warp width for a
  // fifth of the work. Measured, everything after the a0 test is 26.9% of this
  // stage, so running it at true occupancy is worth up to a fifth of the kernel.
  //
  // So survivors are staged instead. Each warp keeps a small shared queue; a
  // ballot gives every live lane a slot, and once 32 have accumulated the warp
  // drains them as one full-width batch. The tail then runs on packed warps
  // rather than mostly-idle ones. Queue depth is 64 because a drain leaves at
  // most 31 behind and the next round can add 32.
  constexpr uint32_t N = seeds_per_thread;
  constexpr uint32_t warps_per_block = threads_per_block / 32;
  constexpr uint32_t stage_depth = 64;

  // [warp][word][slot]: slot last so a batch's lanes read consecutive words.
  // 10 words: post-A-fork rng state, the A yo-fork, and the seed. Carrying the
  // A yo-fork costs shared memory -- 20480 bytes puts this at 4 blocks per SM
  // rather than 8 -- but it beats rebuilding it. A 6-word variant that stages
  // only the pre-fork state and redoes the A fork in the drain fits 7 blocks and
  // still came out slower, -5.42% against -6.77% on this stage, because
  // filter_seeds barely cares about occupancy: forcing 6 blocks costs 0.68% and
  // 5 blocks 1.20%, far less than the fork it would save.
  __shared__ uint32_t stage[warps_per_block][10][stage_depth];

  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  uint32_t staged = 0;   // uniform across the warp

  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t base = start_seed + (uint64_t)index * N;
  const uint64_t S0 = base ^ XrsrRandom::XRSR_SILVER_RATIO;
  const uint64_t X0 = S0 ^ (S0 >> 30);
  uint64_t acc_a = (X0 & ~(uint64_t)(N - 1)) * XrsrRandom::XRSR_MIX1;
  const uint64_t C = S0 ^ (X0 & (N - 1));

#define OMISSION_DRAIN(ACTIVE)                                                         \
  {                                                                                    \
    const uint32_t active_ = (ACTIVE);                                                 \
    if (lane < active_) {                                                              \
      XrsrRandom nr;                                                                   \
      nr.lo = (uint64_t)stage[warp][0][lane] | ((uint64_t)stage[warp][1][lane] << 32);  \
      nr.hi = (uint64_t)stage[warp][2][lane] | ((uint64_t)stage[warp][3][lane] << 32);  \
      XrsrRandomFork a_yo;                                                             \
      a_yo.lo = (uint64_t)stage[warp][4][lane] | ((uint64_t)stage[warp][5][lane] << 32);\
      a_yo.hi = (uint64_t)stage[warp][6][lane] | ((uint64_t)stage[warp][7][lane] << 32);\
      const uint64_t seed_ = (uint64_t)stage[warp][8][lane]                            \
                           | ((uint64_t)stage[warp][9][lane] << 32);                    \
      float sc = 0.35f * fabsf(                                                        \
          octave_yo_mod1<chosen_continentalness_config.octaves_a[0]>(a_yo) - 0.5f);     \
      advance_one(nr);                                                                \
      const auto b_yo = noise_yo_fork(nr.fork());                                       \
      sc += 0.35f * fabsf(                                                              \
          octave_yo_mod1<chosen_continentalness_config.octaves_b[0]>(b_yo) - 0.5f);     \
      if (sc < maxScore) {                                                              \
        sc += 0.11f * fabsf(                                                            \
            octave_yo_mod1<chosen_continentalness_config.octaves_a[1]>(a_yo) - 0.5f);   \
        if (sc < maxScore) {                                                            \
          sc += 0.11f * fabsf(                                                          \
              octave_yo_mod1<chosen_continentalness_config.octaves_b[1]>(b_yo) - 0.5f); \
          if (sc < maxScore) {                                                          \
            sc += 0.04f * fabsf(                                                        \
                octave_yo_mod1<chosen_continentalness_config.octaves_a[2]>(a_yo) - 0.5f);\
            if (sc < maxScore) {                                                        \
              sc += 0.04f * fabsf(                                                      \
                  octave_yo_mod1<chosen_continentalness_config.octaves_b[2]>(b_yo)-0.5f);\
              if (sc < maxScore) {                                                      \
                uint32_t ri = atomicAdd(outputs.len, 1);                                \
                if (ri < outputs.max_len) { outputs.data[ri] = seed_; }                 \
              }                                                                         \
            }                                                                           \
          }                                                                             \
        }                                                                               \
      }                                                                                 \
    }                                                                                   \
  }

#pragma unroll 2
  for (uint32_t r = 0; r < N; ++r, acc_a += XrsrRandom::XRSR_MIX1) {
    const uint64_t s = C ^ r;
    const uint64_t l = mix64_finish(acc_a);
    const uint64_t h = XrsrRandom::mix64(s + XrsrRandom::XRSR_GOLDEN_RATIO);

    const XrsrRandomFork seed_fork = XrsrRandom_seed_fork_from_lh(l, h);
    auto noise_random = seed_fork.from(device_chosen_continentalness_config.fork_hash);
    const auto noise_a_yo_fork = noise_yo_fork(fork_short(noise_random));

    const float c_0A_yo =
        octave_yo_mod1<chosen_continentalness_config.octaves_a[0]>(noise_a_yo_fork);
    const bool live = (0.35f * fabsf(c_0A_yo - 0.5f) < maxScore);

    const uint32_t mask = __ballot_sync(0xFFFFFFFFu, live);
    const uint32_t slot = staged + __popc(mask & ((1u << lane) - 1u));
    if (live) {
      const uint64_t seed_v = s ^ XrsrRandom::XRSR_SILVER_RATIO;
      stage[warp][0][slot] = (uint32_t)noise_random.lo;
      stage[warp][1][slot] = (uint32_t)(noise_random.lo >> 32);
      stage[warp][2][slot] = (uint32_t)noise_random.hi;
      stage[warp][3][slot] = (uint32_t)(noise_random.hi >> 32);
      stage[warp][4][slot] = (uint32_t)noise_a_yo_fork.lo;
      stage[warp][5][slot] = (uint32_t)(noise_a_yo_fork.lo >> 32);
      stage[warp][6][slot] = (uint32_t)noise_a_yo_fork.hi;
      stage[warp][7][slot] = (uint32_t)(noise_a_yo_fork.hi >> 32);
      stage[warp][8][slot] = (uint32_t)seed_v;
      stage[warp][9][slot] = (uint32_t)(seed_v >> 32);
    }
    staged += __popc(mask);

    if (staged >= 32) {
      __syncwarp();
      OMISSION_DRAIN(32u)
      __syncwarp();
      staged -= 32;
      // slide the leftovers down so slot 0 is always the queue head
      if (lane < staged) {
#pragma unroll
        for (uint32_t k = 0; k < 10; ++k) {
          stage[warp][k][lane] = stage[warp][k][lane + 32];
        }
      }
      __syncwarp();
    }
  }

  if (staged != 0) {
    __syncwarp();
    OMISSION_DRAIN(staged)
  }
#undef OMISSION_DRAIN
}

void run(uint64_t start_seed, OutputBuffer<uint64_t> outputs, cudaStream_t stream) {
  const uint64_t aligned_start = start_seed & ~(uint64_t)(seeds_per_thread - 1);
  constexpr uint32_t launch_threads = threads_per_run / seeds_per_thread;
  kernel<<<launch_threads / threads_per_block, threads_per_block, 0, stream>>>(aligned_start, outputs);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilterSeeds

struct SeedPos {
  uint32_t seed_index;
  int32_t x;
  int32_t z;
};

namespace KernelSeed1 {
// The per-iteration survivor ceiling for the whole pipeline. It sizes
// buffer_seeds, buffer_results and buffer_late_init_flags, and KernelFilterSeeds
// silently drops every survivor past it.
//
// Raising it is close to free: this kernel early-exits whole blocks past
// *input.len, so its cost tracks the real survivor count rather than the cap.
// The only real price is buffer_results at sizeof(Result) = 4896 bytes a slot,
// 321 MB at 1<<16 against 641 MB at 1<<17, on a 12 GB card.
//
// It is raised because at 1<<16 the ceiling was quietly distorting the maxScore
// tuning above: 0.042 pinned at exactly 65536 survivors an iteration and 0.045
// pinned harder, so both had their yield truncated and their discovery rate
// understated, which is what made 0.038 look like the peak.
constexpr uint32_t threads_per_run = UINT64_C(1) << 19;
constexpr uint32_t threads_per_block = 32;

struct Result {
  ImprovedNoise continentalness_0A;
  ImprovedNoise continentalness_0B;
  ImprovedNoise continentalness_1A;
  ImprovedNoise continentalness_1B;
  ImprovedNoise continentalness_2A;
  ImprovedNoise continentalness_2B;
  ImprovedNoise continentalness_3A;
  ImprovedNoise continentalness_3B;
  ImprovedNoise continentalness_4A;
  ImprovedNoise continentalness_4B;
  ImprovedNoise continentalness_5A;
  ImprovedNoise continentalness_5B;
  ImprovedNoise continentalness_6A;
  ImprovedNoise continentalness_6B;
  ImprovedNoise continentalness_7A;
  ImprovedNoise continentalness_7B;
  ImprovedNoise continentalness_8A;
  ImprovedNoise continentalness_8B;
};

template <size_t Octaves> struct ResultSampler {
  ImprovedNoise octaves[Octaves];

__device__ float sample_only_a(const GradDotTable &table, int32_t x,
                                 int32_t y, int32_t z) const {
    float val = 0;
    if constexpr (Octaves >= 1)
      val += sample_octave<chosen_continentalness_config.octaves_a[0]>(table, octaves[0], x, y, z);
    if constexpr (Octaves >= 3)
      val += sample_octave<chosen_continentalness_config.octaves_a[1]>(table, octaves[2], x, y, z);
    if constexpr (Octaves >= 5)
      val += sample_octave<chosen_continentalness_config.octaves_a[2]>(table, octaves[4], x, y, z);
    if constexpr (Octaves >= 7)
      val += sample_octave<chosen_continentalness_config.octaves_a[3]>(table, octaves[6], x, y, z);
    if constexpr (Octaves >= 9)
      val += sample_octave<chosen_continentalness_config.octaves_a[4]>(table, octaves[8], x, y, z);
    if constexpr (Octaves >= 11)
      val += sample_octave<chosen_continentalness_config.octaves_a[5]>(table, octaves[10], x, y, z);
    if constexpr (Octaves >= 13)
      val += sample_octave<chosen_continentalness_config.octaves_a[6]>(table, octaves[12], x, y, z);
    if constexpr (Octaves >= 15)
      val += sample_octave<chosen_continentalness_config.octaves_a[7]>(table, octaves[14], x, y, z);
    if constexpr (Octaves >= 17)
      val += sample_octave<chosen_continentalness_config.octaves_a[8]>(table, octaves[16], x, y, z);
    return val;
  }
__device__ float sample(const GradDotTable &table, int32_t x, int32_t y,
                          int32_t z) const {
    float val = 0;
    if constexpr (Octaves >= 1)
      val += sample_octave<chosen_continentalness_config.octaves_a[0]>(table, octaves[0], x, y, z);
    if constexpr (Octaves >= 2)
      val += sample_octave<chosen_continentalness_config.octaves_b[0]>(table, octaves[1], x, y, z);
    if constexpr (Octaves >= 3)
      val += sample_octave<chosen_continentalness_config.octaves_a[1]>(table, octaves[2], x, y, z);
    if constexpr (Octaves >= 4)
      val += sample_octave<chosen_continentalness_config.octaves_b[1]>(table, octaves[3], x, y, z);
    if constexpr (Octaves >= 5)
      val += sample_octave<chosen_continentalness_config.octaves_a[2]>(table, octaves[4], x, y, z);
    if constexpr (Octaves >= 6)
      val += sample_octave<chosen_continentalness_config.octaves_b[2]>(table, octaves[5], x, y, z);
    if constexpr (Octaves >= 7)
      val += sample_octave<chosen_continentalness_config.octaves_a[3]>(table, octaves[6], x, y, z);
    if constexpr (Octaves >= 8)
      val += sample_octave<chosen_continentalness_config.octaves_b[3]>(table, octaves[7], x, y, z);
    if constexpr (Octaves >= 9)
      val += sample_octave<chosen_continentalness_config.octaves_a[4]>(table, octaves[8], x, y, z);
    if constexpr (Octaves >= 10)
      val += sample_octave<chosen_continentalness_config.octaves_b[4]>(table, octaves[9], x, y, z);
    if constexpr (Octaves >= 11)
      val += sample_octave<chosen_continentalness_config.octaves_a[5]>(table, octaves[10], x, y, z);
    if constexpr (Octaves >= 12)
      val += sample_octave<chosen_continentalness_config.octaves_b[5]>(table, octaves[11], x, y, z);
    if constexpr (Octaves >= 13)
      val += sample_octave<chosen_continentalness_config.octaves_a[6]>(table, octaves[12], x, y, z);
    if constexpr (Octaves >= 14)
      val += sample_octave<chosen_continentalness_config.octaves_b[6]>(table, octaves[13], x, y, z);
    if constexpr (Octaves >= 15)
      val += sample_octave<chosen_continentalness_config.octaves_a[7]>(table, octaves[14], x, y, z);
    if constexpr (Octaves >= 16)
      val += sample_octave<chosen_continentalness_config.octaves_b[7]>(table, octaves[15], x, y, z);
    if constexpr (Octaves >= 17)
      val += sample_octave<chosen_continentalness_config.octaves_a[8]>(table, octaves[16], x, y, z);
    if constexpr (Octaves >= 18)
      val += sample_octave<chosen_continentalness_config.octaves_b[8]>(table, octaves[17], x, y, z);
    return val;
  }
};

__device__ void copy_noise(ImprovedNoise (&shared_noise)[threads_per_block], Result *results, ImprovedNoise Result::*result_member, uint32_t block_base, uint32_t input_len) {
  constexpr uint32_t u4_per_struct = sizeof(ImprovedNoise) / sizeof(uint4);
  constexpr uint32_t total_u4 = threads_per_block * u4_per_struct;
  const uint4 *src_flat = reinterpret_cast<const uint4 *>(shared_noise);
  
  for (uint32_t i = threadIdx.x; i < total_u4; i += threads_per_block) {
    uint32_t struct_idx = i / u4_per_struct;
    uint32_t word_idx = i % u4_per_struct;
    
    if (block_base + struct_idx < input_len) {
      ImprovedNoise &dst = results[block_base + struct_idx].*result_member;
      reinterpret_cast<uint4 *>(&dst)[word_idx] = src_flat[i];
    }
  }
}

__device__ void init_octave(const XrsrRandomFork &noise_fork, const XrsrForkHash &fork_hash, Result *results, ImprovedNoise Result::*result_member, uint32_t block_base, uint32_t input_len, bool active) {
  __shared__ alignas(16) ImprovedNoise shared_noise[threads_per_block];

  if (active) {
    init_noise(shared_noise[threadIdx.x], noise_fork.from(fork_hash));
  }
  __syncthreads();

  copy_noise(shared_noise, results, result_member, block_base, input_len);
  __syncthreads();
}

__global__ __launch_bounds__(threads_per_block) void kernel(InputBuffer<uint64_t> input, Result *results) {
  uint32_t block_base = blockIdx.x * blockDim.x;
  uint32_t input_len = *input.len;
  if (block_base >= input_len) {
    return;
  }

  uint32_t index = block_base + threadIdx.x;
  bool active = (index < input_len);

  uint64_t seed = active ? input.data[index] : 0;

  const auto seed_fork = XrsrRandom_seed_fork(seed);
  auto noise_random = seed_fork.from(device_chosen_continentalness_config.fork_hash);

  XrsrRandomFork noise_a_fork, noise_b_fork;
  XrsrRandom_double_fork(noise_random, noise_a_fork, noise_b_fork);

  init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[0].fork_hash, results, &Result::continentalness_0A, block_base, input_len, active);
  init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[1].fork_hash, results, &Result::continentalness_1A, block_base, input_len, active);

  // 0B and 1B are deliberately NOT built here. init_seeds runs for every seed
  // that clears filter_seeds -- 42706 an iteration -- but nothing before
  // gradvecs_2 reads the B stack: gradvecs_1 samples 0A, and filter_2_01a is
  // only_a, so it takes 0A and 1A. gradvecs_2 needs 0B and filter_2_01b needs
  // both 0B and 1B (it is a hand-written specialisation that samples
  // octaves_b[0] and octaves_b[1], not the generic three-octave path). Those two
  // stages see 296028 candidates an iteration between them, so building the B
  // stack here was a large overbuild; run_late<1> builds it lazily instead.
}

__device__ void copy_noise_direct(const ImprovedNoise &shared_noise, Result *results, ImprovedNoise Result::*result_member, uint32_t seed_index) {
  constexpr uint32_t u4_per_struct = sizeof(ImprovedNoise) / sizeof(uint4);
  ImprovedNoise &dst = results[seed_index].*result_member;
  const uint4 *src = reinterpret_cast<const uint4 *>(&shared_noise);
  uint4 *dst_words = reinterpret_cast<uint4 *>(&dst);
#pragma unroll
  for (uint32_t word_idx = 0; word_idx < u4_per_struct; word_idx++) {
    dst_words[word_idx] = src[word_idx];
  }
}

__device__ void init_octave_direct(ImprovedNoise &shared_noise, const XrsrRandomFork &noise_fork, const XrsrForkHash &fork_hash, Result *results, ImprovedNoise Result::*result_member, uint32_t seed_index) {
  init_noise(shared_noise, noise_fork.from(fork_hash));
  copy_noise_direct(shared_noise, results, result_member, seed_index);
}

template <uint32_t Stage>
__global__ __launch_bounds__(threads_per_block) void late_kernel(InputBuffer<uint64_t> seeds, InputBuffer<SeedPos> inputs, Result *results, uint32_t *init_flags) {
  __shared__ alignas(16) ImprovedNoise shared_noise[threads_per_block];

  const uint32_t inputs_len = *inputs.len;
  for (uint32_t input_index = blockIdx.x * blockDim.x + threadIdx.x; input_index < inputs_len; input_index += gridDim.x * blockDim.x) {
    const uint32_t seed_index = inputs.data[input_index].seed_index;
    if (atomicCAS(&init_flags[seed_index], Stage - 1u, Stage) != Stage - 1u) {
      continue;
    }

    const uint64_t seed = seeds.data[seed_index];
    const auto seed_fork = XrsrRandom_seed_fork(seed);
    auto noise_random = seed_fork.from(device_chosen_continentalness_config.fork_hash);

    XrsrRandomFork noise_a_fork, noise_b_fork;
    XrsrRandom_double_fork(noise_random, noise_a_fork, noise_b_fork);

    ImprovedNoise &noise = shared_noise[threadIdx.x];
    if constexpr (Stage == 1) {
    // The B stack, for gradvecs_2 and filter_2_01b. Both must land here: 1B is
    // read by filter_2_01b, so deferring it to the next stage is too late.
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[0].fork_hash, results, &Result::continentalness_0B, seed_index);
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[1].fork_hash, results, &Result::continentalness_1B, seed_index);
    } else if constexpr (Stage == 2) {
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[2].fork_hash, results, &Result::continentalness_2A, seed_index);
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[3].fork_hash, results, &Result::continentalness_3A, seed_index);
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[4].fork_hash, results, &Result::continentalness_4A, seed_index);
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[5].fork_hash, results, &Result::continentalness_5A, seed_index);

    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[2].fork_hash, results, &Result::continentalness_2B, seed_index);
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[3].fork_hash, results, &Result::continentalness_3B, seed_index);
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[4].fork_hash, results, &Result::continentalness_4B, seed_index);
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[5].fork_hash, results, &Result::continentalness_5B, seed_index);
    } else if constexpr (Stage == 3) {
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[6].fork_hash, results, &Result::continentalness_6A, seed_index);
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[6].fork_hash, results, &Result::continentalness_6B, seed_index);
    } else if constexpr (Stage == 4) {
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[7].fork_hash, results, &Result::continentalness_7A, seed_index);
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[7].fork_hash, results, &Result::continentalness_7B, seed_index);
    } else if constexpr (Stage == 5) {
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[8].fork_hash, results, &Result::continentalness_8A, seed_index);
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[8].fork_hash, results, &Result::continentalness_8B, seed_index);
    }
  }
}

template <uint32_t Stage>
void run_late(InputBuffer<uint64_t> seeds, InputBuffer<SeedPos> inputs, Result *results, uint32_t *init_flags, cudaStream_t stream) {
  late_kernel<Stage><<<1024, threads_per_block, 0, stream>>>(seeds, inputs, results, init_flags);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelSeed1

constexpr int32_t large_biomes_pos_mul = large_biomes ? 4 : 1;

#include "kernel_0A.h"
__device__ float device_kernel_0A[6][6][16][2];
static_assert(sizeof(host_kernel_0A) == sizeof(device_kernel_0A));

#include "kernel_0B.h"
__device__ float device_kernel_0B[6][6][16][2];
static_assert(sizeof(host_kernel_0B) == sizeof(device_kernel_0B));

void init_conv_kernels() {
  float temp_0A[6][6][16][2];
  for (int dny = 0; dny < 2; ++dny) {
    for (int dnx = 0; dnx < 6; ++dnx) {
      for (int dnz = 0; dnz < 6; ++dnz) {
        for (int p = 0; p < 16; ++p) {
          temp_0A[dnx][dnz][p][dny] = host_kernel_0A[dny][dnx][dnz][p];
        }
      }
    }
  }
  void *device_kernel_0A_addr;
  TRY_CUDA(cudaGetSymbolAddress(&device_kernel_0A_addr, device_kernel_0A));
  TRY_CUDA(cudaMemcpy(device_kernel_0A_addr, temp_0A, sizeof(temp_0A), cudaMemcpyHostToDevice));
  float temp_0B[6][6][16][2];
  for (int dny = 0; dny < 2; ++dny) {
    for (int dnx = 0; dnx < 6; ++dnx) {
      for (int dnz = 0; dnz < 6; ++dnz) {
        for (int p = 0; p < 16; ++p) {
          temp_0B[dnx][dnz][p][dny] = host_kernel_0B[dny][dnx][dnz][p];
        }
      }
    }
  }
  void *device_kernel_0B_addr;
  TRY_CUDA(cudaGetSymbolAddress(&device_kernel_0B_addr, device_kernel_0B));
  TRY_CUDA(cudaMemcpy(device_kernel_0B_addr, temp_0B, sizeof(temp_0B), cudaMemcpyHostToDevice));
}

// Two-stage gate. The 2x2 prefilter reads two values from conv_z0 and two from
// conv_z1; a gate-disabled probe showed it is ~93% of KernelFilterGradVecs1 and
// the kernel runs near this device's shared-memory roofline, so those loads are
// the binding cost. Splitting it lets the conv_z1 half be skipped for the large
// majority of candidates: only the z0 pair is read unconditionally, and the z1
// pair is fetched when the z0 half alone clears kGradVecs1GateZ0Threshold.
// This is deliberately lossy - a candidate whose z1 contribution would have
// carried it over the line is dropped. At -4.0 three of the ten reference seeds
// in seeds.txt are no longer found, including a 12,336,736-block island; -5.0
// keeps all ten and is about 7% slower. What the threshold drops is marginal
// detections rather than small islands: the seeds that disappear are the ones
// found only once at baseline, and island size does not predict survival.
// The 6x6 matched filter is far sparser than it looks. Taking the variance of
// the 16 gradient values at each (x,z) cell of kernel_0A, 92% of the total sits
// in the central 2x2 block, 96% in x-columns 2 and 3, and 99.6% in columns 1-4.
// Columns 0 and 5 are very nearly constant in the gradient index and so carry
// almost no information about a candidate.
//
// Keep columns 1..4 and fold the discarded pair into the threshold: they have
// mean -1.9953 and sd 0.269, so testing the 8-term sum against
// (-18.0 - -1.9953) = -16.0047 reproduces the old 12-term test to within 0.3%
// of the score variance. That removes the separate 12-term pass entirely -- the
// prefilter gate and the score now read the same table -- and drops the kernel
// from 40 registers / 17712 B of shared to 32 / 13616.
//
// Keeping only columns 2 and 3 was also tried: faster still, but discarding 4%
// of the variance (sd 0.939) costs marginal detections, losing two more of the
// ten reference seeds even when the threshold is loosened to match yield.
//
// Predicted mean/sd of the score from this decomposition are -34.277 / 4.708;
// measured over 42630 seeds they are -34.28 / 4.69.
constexpr float kGradVecs1GateZ0Threshold = -4.0f;

// This gate is the hottest read in the program, and it is bound by instruction
// issue, not by shared-memory bandwidth. The distinction was measured.
//
// At a row stride of 6 floats the fused LDS.U.64 reaches only 16 of the 32
// shared banks (gcd(6,32) = 2), so every gate load is a 2-way conflict. Packing
// the two columns into 16-bit fixed point (bias 12, scale 2730, both columns
// then fitting in [0, 32760] so their sum can be compared in the high half of a
// word) halves the bytes moved and makes the access conflict-free at stride 1.
// Built and measured, it was 23% slower on this kernel and 9.7% slower overall,
// bit-exact -- the four integer ops it needs per candidate cost more than the
// bandwidth it saves, because the conflict replays already hide under the ALU
// work. The float pair plus one FADD and one FSETP is the cheaper formulation.
//
// A 4-way conflict is a different matter: at stride 4 the same loads reach only
// 8 banks and the kernel turns 54% slower, so the LSU does become the limit
// there. This sits just on the ALU-bound side of that crossover.
constexpr float kGradVecs1PrefilterThreshold = -12.0f;
constexpr float kGradVecs1FinalThreshold      = -16.0047f;  // 8-term score, see above

constexpr float kGradVecs2PrefilterThreshold = -13.5f;
constexpr float kGradVecs2FinalThreshold      = -20.0f;

// Layout note: KernelFilterGradVecs1 keeps conv as [256][6]. Consecutive
// candidates reuse the same permutation index at adjacent columns, so ptxas
// fuses those pairs into LDS.U.64, and with idx = base + threadIdx.x the two
// 16-lane phases of that access cover all 32 banks exactly once. Transposing it
// measurably loses. KernelFilterGradVecs2 has no such sliding-window reuse, so
// it uses the transposed [6][256] layout, which is conflict-free there.
template <typename IndexT>
__device__ __forceinline__ float score_center_2x2(
    const float conv_z0[256][6],
    const float conv_z1[256][6],
    const IndexT* idx0,
    const IndexT* idx1)
{
  return
      conv_z0[idx0[2] & 0xFF][2] +
      conv_z0[idx0[3] & 0xFF][3] +
      conv_z1[idx1[2] & 0xFF][2] +
      conv_z1[idx1[3] & 0xFF][3];
}

template <typename IndexT>
__device__ __forceinline__ float score_center_z0(
    const float conv_z0[256][6], const IndexT* idx0)
{
  return conv_z0[idx0[2] & 0xFF][2] + conv_z0[idx0[3] & 0xFF][3];
}

template <typename IndexT>
__device__ __forceinline__ float score_center_z1(
    const float conv_z1[256][6], const IndexT* idx1)
{
  return conv_z1[idx1[2] & 0xFF][2] + conv_z1[idx1[3] & 0xFF][3];
}

template <typename IndexT>
__device__ __forceinline__ float score_center_2x2_t(
    const float conv_z0[6][256],
    const float conv_z1[6][256],
    const IndexT* idx0,
    const IndexT* idx1)
{
  return
      conv_z0[2][idx0[2] & 0xFF] +
      conv_z0[3][idx0[3] & 0xFF] +
      conv_z1[2][idx1[2] & 0xFF] +
      conv_z1[3][idx1[3] & 0xFF];
}

template <typename IndexT>
__device__ __forceinline__ float score_full_12(
    const float conv_z0[256][6],
    const float conv_z1[256][6],
    const IndexT* idx0,
    const IndexT* idx1)
{
  float score = 0.0f;
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    score += conv_z0[idx0[i] & 0xFF][i];
    score += conv_z1[idx1[i] & 0xFF][i];
  }
  return score;
}

template <typename IndexT>
__device__ __forceinline__ float score_full_12_t(
    const float conv_z0[6][256],
    const float conv_z1[6][256],
    const IndexT* idx0,
    const IndexT* idx1)
{
  float score = 0.0f;
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    score += conv_z0[i][idx0[i] & 0xFF];
    score += conv_z1[i][idx1[i] & 0xFF];
  }
  return score;
}

namespace KernelFilterGradVecs1 {
constexpr uint32_t block_dim_x = 256;

__global__
__launch_bounds__(block_dim_x) void kernel(
    const InputBuffer<uint64_t> seeds,
    OutputBuffer<SeedPos> outputs,
    const KernelSeed1::Result* __restrict__ results)
{
  __shared__ alignas(16) ImprovedNoise oct_0A;
  // Occupancy is at its optimum here and should be left alone. This block uses
  // 17712 bytes of shared memory, which fits 5 blocks per SM (62% occupancy).
  // Both neighbours were built and measured round-robin against it:
  //
  //   4 blocks / 50%  (padded to 20112 B)              gradvecs_1  +4.18%
  //   5 blocks / 62%  (this)                           gradvecs_1   best
  //   6 blocks / 75%  (staging only kernel columns 1-4
  //                    for 16176 B, under the 16384
  //                    needed for a 6th block)         gradvecs_1  +1.23%
  //
  // Dropping to 4 hurts, but buying a 6th block does not help -- the kernel is
  // issue-bound, so extra resident warps have no latency left to hide, and the
  // narrower staging costs a little address arithmetic. 5 is the peak.
  __shared__ alignas(16) float shared_kernel_0A[6][6][16][2];

  // only columns x = 1..4 carry information, but the row stride stays 6: at
  // stride 6 the gate pair (x = 2, 3) sits at float offset 6i+2, always 8-byte
  // aligned, so ptxas fuses the two loads into one LDS.U.64. A stride of 4 puts
  // them at 4i+1 / 4i+2, never 8-byte aligned, which costs 54% on this kernel.
  // Only kernel columns 1..4 are ever read, so the row is 5 wide, not 6, with
  // column x stored at x-1 and one slot of padding. Two gains: 2048 fewer bytes
  // of shared memory, and a stride of 5, which is coprime with the 32 shared
  // banks -- since the candidate list made these reads random rather than
  // consecutive, an odd stride reaches all 32 banks where 6 reached only 16.
  //
  // A tighter gate was tried on top of this and rejected. The bound here is
  // one-sided: rows survive if A[t] >= T - max(B). Adding the mirror condition
  // B[s] >= T - max(A) and keeping only pairs whose difference matches some nx's
  // lag cuts the work from 65536 grid points to about 870 pairs -- 75x fewer,
  // and 8.5x fewer than the one-sided list's 7398 checks. Both a prefix-summed
  // CSR of nx-by-lag and fixed-capacity buckets were built, both bit-exact at
  // 49137561 candidates, and both were worth almost nothing: -5.8% and -8.1% on
  // this kernel against a predicted 3.6x. gradvecs_1 sees only ~4.5 hits per
  // seed, so once the scan is down to 28 rows the per-seed bookkeeping -- two
  // block-wide max reductions, two ballot compactions, the lag structure and its
  // barriers -- costs more than the scan it removes. The one-sided list already
  // takes essentially all of the available win.
  __shared__ float conv_z0[256][5];
  __shared__ float conv_z1[256][5];

  __shared__ alignas(16) uint8_t idx_xy[2][272];
  // The gate is A[t] + B[t+d] >= T with A = conv_z0[.][2] and B = conv_z0[.][3].
  // B can never exceed max(B), so only t with A[t] >= T - max(B) can contribute
  // to any candidate at all. Measured over random permutations, that is 11% of
  // the 256 t values, so listing them cuts gate checks about ninefold with no
  // loss -- every (t, d) that really passes is kept, verified exhaustively.
  __shared__ uint8_t cand_t[256];
  __shared__ float   cand_a[256];
  __shared__ uint32_t cand_n;
  // Queue of gate survivors awaiting their score, one per warp. The gate lets
  // roughly 166 of a seed's 65536 positions through, scattered across the scan,
  // so a warp reaching the score path carries less than one live lane on
  // average -- measured, the score path is 69% of this kernel while the scan
  // that feeds it is 1%. Staging the survivors and draining 32 at a time runs
  // that work at full width instead. Depth 64: a drain leaves at most 31 behind
  // and the next round can add 32.
  __shared__ uint32_t score_q[block_dim_x / 32][64];
  __shared__ float   red_max[8];

  const int32_t nz = threadIdx.x;

  for (uint32_t i = nz; i < 288; i += block_dim_x) {
    reinterpret_cast<uint4*>(shared_kernel_0A)[i] =
        reinterpret_cast<const uint4*>(device_kernel_0A)[i];
  }

  const uint32_t seeds_len = *seeds.len;
  for (uint32_t seed_index = blockIdx.x; seed_index < seeds_len; seed_index += gridDim.x) {
    __syncthreads();

    if (nz < 17) {
      reinterpret_cast<uint4*>(&oct_0A)[nz] =
          reinterpret_cast<const uint4*>(&results[seed_index].continentalness_0A)[nz];
    }

    __syncthreads();

    {
      uint32_t p_z[6];
#pragma unroll
      for (int32_t dnz = 0; dnz < 6; ++dnz) {
        p_z[dnz] = oct_0A.p[(nz + dnz) & 0xFF] & 0xF;
      }

#pragma unroll
      for (int32_t dnx = 1; dnx < 5; ++dnx) {
        float conv0 = 0.0f;
        float conv1 = 0.0f;

#pragma unroll
        for (int32_t dnz = 0; dnz < 6; ++dnz) {
          const uint32_t p = p_z[dnz];
          conv0 += shared_kernel_0A[dnx][dnz][p][0];
          conv1 += shared_kernel_0A[dnx][dnz][p][1];
        }

        conv_z0[nz][dnx - 1] = conv0;
        conv_z1[nz][dnx - 1] = conv1;
      }
    }

    const int32_t cell_size = 512 * large_biomes_pos_mul;
    const int32_t x_center = (2.5f - oct_0A.xo) * cell_size;
    const int32_t ny = oct_0A.yo;
    const int32_t z_center = (2.5f - oct_0A.zo) * cell_size;

    const uint8_t idx_x = oct_0A.p[nz];
    const uint8_t v0 = oct_0A.p[(idx_x + ny) & 0xFF];
    const uint8_t v1 = oct_0A.p[(idx_x + ny + 1) & 0xFF];
    idx_xy[0][nz] = v0;
    idx_xy[1][nz] = v1;

    if (nz < 6) {
      idx_xy[0][256 + nz] = v0;
      idx_xy[1][256 + nz] = v1;
    }

    if (nz == 0) { cand_n = 0; }
    __syncthreads();

    // max over B = conv_z0[.][3], reduced across the block
    {
      float m = conv_z0[nz][2];
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        m = fmaxf(m, __shfl_down_sync(0xFFFFFFFFu, m, off));
      }
      if ((nz & 31) == 0) { red_max[nz >> 5] = m; }
    }
    __syncthreads();
    float max_b = red_max[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) { max_b = fmaxf(max_b, red_max[k]); }

    // list the t values that could still clear the gate
    {
      const float a = conv_z0[nz][1];
      // Slack, because the bound is applied in floating point: fl(a + b) can
      // reach the threshold when a sits just under fl(T - max_b). Values here
      // are order 10, so an ulp is ~1e-6; 1e-3 is far more than enough and
      // widens the list by a negligible amount.
      const bool keep = (a >= kGradVecs1GateZ0Threshold - max_b - 1.0e-3f);
      const uint32_t m = __ballot_sync(0xFFFFFFFFu, keep);
      uint32_t base;
      if ((nz & 31) == 0) { base = atomicAdd(&cand_n, __popc(m)); }
      base = __shfl_sync(0xFFFFFFFFu, base, 0);
      if (keep) {
        const uint32_t slot = base + __popc(m & ((1u << (nz & 31)) - 1u));
        cand_t[slot] = (uint8_t)nz;
        cand_a[slot] = a;
      }
    }
    __syncthreads();
    const uint32_t n_cand = cand_n;

    // One thread per nx now, rather than per nz: nx fixes the four y=0 and four
    // y=1 lattice offsets for the whole seed, so they are read once into
    // registers instead of being re-derived by a sliding window.
    const uint32_t nx = nz;
    const uint32_t warp = nz >> 5, lane = nz & 31;
    const uint32_t p1 = idx_xy[0][nx + 1], p2 = idx_xy[0][nx + 2];
    const uint32_t p3 = idx_xy[0][nx + 3], p4 = idx_xy[0][nx + 4];
    const uint32_t q1 = idx_xy[1][nx + 1], q2 = idx_xy[1][nx + 2];
    const uint32_t q3 = idx_xy[1][nx + 3], q4 = idx_xy[1][nx + 4];
    const uint32_t d = (p3 - p2) & 0xFF;
    const int32_t x = x_center + (int32_t)nx * cell_size;

#define OMISSION_SCORE_DRAIN(ACTIVE)                                             \
    {                                                                            \
      const uint32_t act_ = (ACTIVE);                                            \
      if (lane < act_) {                                                         \
        const uint32_t e_ = score_q[warp][lane];                                 \
        const uint32_t nxv = e_ >> 8, t_ = e_ & 0xFFu;                           \
        const uint32_t r1 = idx_xy[0][nxv + 1], r2 = idx_xy[0][nxv + 2];         \
        const uint32_t r3 = idx_xy[0][nxv + 3], r4 = idx_xy[0][nxv + 4];         \
        const uint32_t s1 = idx_xy[1][nxv + 1], s2 = idx_xy[1][nxv + 2];         \
        const uint32_t s3 = idx_xy[1][nxv + 3], s4 = idx_xy[1][nxv + 4];         \
        const uint32_t nzv = (t_ - r2) & 0xFF;                                   \
        const float sc = (conv_z0[(r2 + nzv) & 0xFF][1]                         \
                        + conv_z0[(r3 + nzv) & 0xFF][2])                        \
                       + conv_z0[(r1 + nzv) & 0xFF][0]                          \
                       + conv_z0[(r4 + nzv) & 0xFF][3]                          \
                       + conv_z1[(s1 + nzv) & 0xFF][0]                          \
                       + conv_z1[(s2 + nzv) & 0xFF][1]                          \
                       + conv_z1[(s3 + nzv) & 0xFF][2]                          \
                       + conv_z1[(s4 + nzv) & 0xFF][3];                          \
        if (sc > kGradVecs1FinalThreshold) {                                     \
          uint32_t ri_ = atomicAdd(outputs.len, 1);                              \
          if (ri_ < outputs.max_len) {                                           \
            outputs.data[ri_] = {seed_index,                                     \
                                 x_center + (int32_t)nxv * cell_size,            \
                                 z_center + (int32_t)nzv * cell_size};           \
          }                                                                      \
        }                                                                        \
      }                                                                          \
    }

    uint32_t sq_n = 0;   // uniform across the warp
    for (uint32_t i = 0; i < n_cand; ++i) {
      const uint32_t t = cand_t[i];          // broadcast
      const float    a = cand_a[i];          // broadcast
      const float    b = conv_z0[(t + d) & 0xFF][2];
      const bool hit = (a + b >= kGradVecs1GateZ0Threshold);

      const uint32_t m = __ballot_sync(0xFFFFFFFFu, hit);
      if (hit) {
        score_q[warp][sq_n + __popc(m & ((1u << lane) - 1u))] = (nx << 8) | t;
      }
      sq_n += __popc(m);

      if (sq_n >= 32u) {
        __syncwarp();
        OMISSION_SCORE_DRAIN(32u)
        __syncwarp();
        sq_n -= 32u;
        if (lane < sq_n) { score_q[warp][lane] = score_q[warp][lane + 32]; }
        __syncwarp();
      }
    }

    if (sq_n != 0u) {
      __syncwarp();
      OMISSION_SCORE_DRAIN(sq_n)
    }
#undef OMISSION_SCORE_DRAIN
  }
}

void run(
    const InputBuffer<uint64_t> seeds,
    OutputBuffer<SeedPos> outputs,
    const KernelSeed1::Result* __restrict__ results,
    cudaStream_t stream)
{
  // 4096 blocks rather than 2048: with 5 blocks per SM resident this deepens
  // the queue of work available to hide the per-seed global load and the
  // barriers around the convolution build. Measured -1.9% on this stage; 8192
  // measures the same and 1024 is 2.7% worse.
  kernel<<<4096, block_dim_x, 0, stream>>>(seeds, outputs, results);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilterGradVecs1

namespace KernelFilterGradVecs2 {
constexpr uint32_t block_dim_x = 128;
constexpr uint32_t grid_width = unbound ? 331 : (large_biomes ? 29 : 115); //checking full bounded world / full unbounded
constexpr uint32_t threads_per_seed = grid_width * grid_width;
constexpr uint32_t grid_half = grid_width / 2;

__global__
__launch_bounds__(block_dim_x) void kernel(
    InputBuffer<SeedPos> inputs,
    OutputBuffer<SeedPos> outputs,
    const KernelSeed1::Result* __restrict__ results)
{
  __shared__ alignas(16) ImprovedNoise oct_0B;
  __shared__ alignas(16) float shared_kernel_0B[6][6][16][2];

  __shared__ float conv_z0[6][256];
  __shared__ float conv_z1[6][256];

  for (uint32_t i = threadIdx.x; i < 288; i += blockDim.x) {
    reinterpret_cast<uint4*>(shared_kernel_0B)[i] =
        reinterpret_cast<const uint4*>(device_kernel_0B)[i];
  }

  constexpr int32_t cell_size_0A = (int32_t)(1.0f / chosen_continentalness_config.octaves_a[0].input_factor) * 256;
  const float input_factor_b = chosen_continentalness_config.octaves_b[0].input_factor;
  const int32_t grid_half_s = (int32_t)grid_half;

  const uint32_t inputs_len = *inputs.len;
  for (uint32_t input_index = blockIdx.x; input_index < inputs_len; input_index += gridDim.x) {
    const SeedPos input = inputs.data[input_index];

    __syncthreads();

    if (threadIdx.x < 17) {
      reinterpret_cast<uint4*>(&oct_0B)[threadIdx.x] =
          reinterpret_cast<const uint4*>(&results[input.seed_index].continentalness_0B)[threadIdx.x];
    }

    __syncthreads();

    for (int32_t V = threadIdx.x; V < 256; V += blockDim.x) {
      uint32_t p_z[6];
#pragma unroll
      for (int32_t dnz = 0; dnz < 6; ++dnz) {
        p_z[dnz] = oct_0B.p[(V + dnz) & 0xFF] & 0xF;
      }

#pragma unroll
      for (int32_t dnx = 0; dnx < 6; ++dnx) {
        float conv0 = 0.0f;
        float conv1 = 0.0f;

#pragma unroll
        for (int32_t dnz = 0; dnz < 6; ++dnz) {
          const uint32_t p = p_z[dnz];
          conv0 += shared_kernel_0B[dnx][dnz][p][0];
          conv1 += shared_kernel_0B[dnx][dnz][p][1];
        }

        conv_z0[dnx][V] = conv0;
        conv_z1[dnx][V] = conv1;
      }
    }

    __syncthreads();

    const int32_t ny = oct_0B.yo;

    for (uint32_t tx = 0; tx < grid_width; ++tx) {
      const int32_t tile_dx = ((int32_t)tx - grid_half_s) * cell_size_0A;
      const int32_t x = input.x + tile_dx;
      const int32_t nx = __float2int_rd(x * input_factor_b + oct_0B.xo - 2.0f);

      int32_t hoisted_idx_xy[2][6];
#pragma unroll
      for (int32_t dnx = 0; dnx < 6; ++dnx) {
        const int32_t idx_x = oct_0B.p[(nx + dnx) & 0xFF];
#pragma unroll
        for (int32_t dny = 0; dny < 2; ++dny) {
          hoisted_idx_xy[dny][dnx] = oct_0B.p[(idx_x + ny + dny) & 0xFF];
        }
      }

      for (uint32_t tz = threadIdx.x; tz < grid_width; tz += blockDim.x) {
        const int32_t tile_dz = ((int32_t)tz - grid_half_s) * cell_size_0A;
        const int32_t z = input.z + tile_dz;

        const int32_t nz = __float2int_rd(z * input_factor_b + oct_0B.zo - 2.0f);
        const int32_t nz_masked = nz & 0xFF;

        int32_t idx0[6];
        int32_t idx1[6];
#pragma unroll
        for (int32_t i = 0; i < 6; ++i) {
          idx0[i] = hoisted_idx_xy[0][i] + nz_masked;
          idx1[i] = hoisted_idx_xy[1][i] + nz_masked;
        }

        const float gate = score_center_2x2_t(conv_z0, conv_z1, idx0, idx1);
        if (gate >= kGradVecs2PrefilterThreshold) {
          const float score = score_full_12_t(conv_z0, conv_z1, idx0, idx1);
          if (score > kGradVecs2FinalThreshold) {
            uint32_t result_index = atomicAdd(outputs.len, 1);
            if (result_index < outputs.max_len) {
              outputs.data[result_index] = {input.seed_index, x, z};
            }
          }
        }
      }
    }
  }
}

void run(
    const InputBuffer<SeedPos> inputs,
    OutputBuffer<SeedPos> outputs,
    const KernelSeed1::Result* __restrict__ results,
    cudaStream_t stream)
{
  kernel<<<2048, block_dim_x, 0, stream>>>(inputs, outputs, results);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilterGradVecs2

namespace KernelFilter1 {
constexpr uint32_t threads_per_block = 256;
constexpr uint32_t threads_per_seed_sqrt = UINT64_C(1) << 10;
constexpr uint32_t threads_per_seed = threads_per_seed_sqrt * threads_per_seed_sqrt;
// noise (1:4) coords
constexpr int32_t pos_step = 14600 * large_biomes_pos_mul / 4;
constexpr int32_t pos_range = (int32_t)threads_per_seed_sqrt * pos_step;
static_assert(pos_range <= 60'000'000 / 4);

__global__ __launch_bounds__(threads_per_block) void kernel(InputBuffer<uint64_t> seeds, OutputBuffer<SeedPos> outputs, KernelSeed1::Result *results) {
  __shared__ GradDotTable shared_grad_dot_table;
  if (threadIdx.x < sizeof(shared_grad_dot_table) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&shared_grad_dot_table)[threadIdx.x] = reinterpret_cast<uint32_t *>(&device_grad_dot_table)[threadIdx.x];
  }

  uint32_t seeds_len = *seeds.len;

  uint64_t total_threads = (uint64_t)seeds_len * threads_per_seed;
  for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < total_threads; index += (uint64_t)gridDim.x * blockDim.x) {
    uint32_t seed_index = index / threads_per_seed;
    uint32_t pos_index = index % threads_per_seed;

    uint32_t x_index = pos_index % threads_per_seed_sqrt;
    uint32_t z_index = pos_index / threads_per_seed_sqrt;

    int32_t x = (int32_t)x_index * pos_step - pos_range / 2;
    int32_t z = (int32_t)z_index * pos_step - pos_range / 2;

    // no more smem
    const auto &octaves = reinterpret_cast<const KernelSeed1::ResultSampler<2> &>(results[seed_index]);

    float val = octaves.sample(shared_grad_dot_table, x, 0, z);

    if (val >= -0.515f)
      continue; // 1 in 27.7

    uint32_t result_index = atomicAdd(outputs.len, 1);
    if (result_index >= outputs.max_len){
      continue;
    }
    outputs.data[result_index] = {seed_index, x, z};
  }
}

void run(InputBuffer<uint64_t> seeds, OutputBuffer<SeedPos> outputs, KernelSeed1::Result *results, cudaStream_t stream) {
  kernel<<<16 * 1024, threads_per_block, 0, stream>>>(seeds, outputs, results);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilter1

constexpr bool is_pow2(uint32_t val) { return (val & (val - 1)) == 0; }

constexpr uint32_t log2(uint32_t val) { return 31 - std::countl_zero(val); }

template <typename T> __device__ T warp_reduce_add(T val) {
#if __CUDA_ARCH__ >= 800
  return __reduce_add_sync(0xFFFFFFFF, val);
#else
  val += __shfl_down_sync(-1u, val, 1);
  val += __shfl_down_sync(-1u, val, 2);
  val += __shfl_down_sync(-1u, val, 4);
  val += __shfl_down_sync(-1u, val, 8);
  val += __shfl_down_sync(-1u, val, 16);
  return val;
#endif
}

namespace KernelFilter2 {
template <int32_t NoiseThreshold, size_t Octaves, uint32_t PosRange, uint32_t Samples, uint32_t MinCount, bool FlippedSparseSamples, bool MoveCenter, bool OnlyA>
struct Template {
  static constexpr float noise_threshold = NoiseThreshold / 10000.0f;
  static constexpr size_t octaves = Octaves;
  static constexpr uint32_t pos_range = PosRange;
  static constexpr uint32_t samples = Samples;
  static constexpr uint32_t min_count = MinCount;
  static constexpr bool flipped_sparse_samples = FlippedSparseSamples;
  static constexpr bool move_center = MoveCenter;
  static constexpr bool only_a = OnlyA;

  static constexpr uint32_t threads_per_block = 256;
  static_assert(samples >= 32 && samples <= threads_per_block * threads_per_block && is_pow2(samples));
  static constexpr uint32_t samples_square_size = UINT32_C(1) << (log2(samples) + 1) / 2;
  static constexpr bool samples_square_sparse = log2(samples) % 2 == 1;
  static_assert(!flipped_sparse_samples || samples_square_sparse);
  static_assert(pos_range * large_biomes_pos_mul % (samples_square_size * 2 * 4) == 0);
  
  static constexpr uint32_t pos_step = pos_range * large_biomes_pos_mul / 4 / samples_square_size;
  static constexpr int32_t pos_offset = -(int32_t)(pos_step * (samples_square_size - 1) / 2);

  static constexpr uint32_t threads_per_input = std::min(samples, threads_per_block);
  static constexpr uint32_t loops = samples / threads_per_input;
  static constexpr uint32_t inputs_per_block = threads_per_block / threads_per_input;

  static void run(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs, KernelSeed1::Result *results, cudaStream_t stream);
};

template <typename T>
__global__ __launch_bounds__(T::threads_per_block) void kernel(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs, KernelSeed1::Result *results) {
  __shared__ GradDotTable shared_grad_dot_table;
  
  constexpr uint32_t grad_table_words = sizeof(GradDotTable) / sizeof(uint32_t);
  for (uint32_t i = threadIdx.x; i < grad_table_words; i += blockDim.x) {
    reinterpret_cast<uint32_t *>(&shared_grad_dot_table)[i] = reinterpret_cast<uint32_t *>(&device_grad_dot_table)[i];
  }
  __syncthreads();

  const uint32_t inputs_len = *inputs.len;
  
  constexpr uint32_t threads_per_input = T::threads_per_input;
  constexpr uint32_t inputs_per_block = T::inputs_per_block;
  
  const uint32_t block_input_index = threadIdx.x / threads_per_input;
  const uint32_t pos_index = threadIdx.x % threads_per_input;

  constexpr int32_t z_step = (int32_t)(T::pos_step * (T::samples_square_size / T::loops));

  __shared__ uint32_t shared_counts[inputs_per_block];
  __shared__ int32_t shared_sums[inputs_per_block][2];

  for (uint32_t block_input_base = blockIdx.x * inputs_per_block; 
       block_input_base < inputs_len; 
       block_input_base += gridDim.x * inputs_per_block) {
    
    const uint32_t input_index = block_input_base + block_input_index;
    const bool is_valid_input = (input_index < inputs_len);

    if constexpr (T::samples > 32) {
      if (threadIdx.x < inputs_per_block) {
        shared_counts[threadIdx.x] = 0;
        if constexpr (T::move_center) {
          shared_sums[threadIdx.x][0] = 0;
          shared_sums[threadIdx.x][1] = 0;
        }
      }
      __syncthreads();
    }

    uint32_t total_valid = 0;
    int32_t sum_dx = 0;
    int32_t sum_dz = 0;
    SeedPos input = {};

    if (is_valid_input) {
      input = inputs.data[input_index];

      const uint32_t x_index = pos_index % T::samples_square_size;
      uint32_t z_index = pos_index / T::samples_square_size;
      if constexpr (T::samples_square_sparse) {
        z_index = z_index * 2 + ((x_index & 1) ^ T::flipped_sparse_samples);
      }

      int32_t x = input.x + (int32_t)(x_index * T::pos_step) + T::pos_offset;
      int32_t z = input.z + (int32_t)(z_index * T::pos_step) + T::pos_offset;

      const auto &octaves = reinterpret_cast<const KernelSeed1::ResultSampler<T::octaves> &>(results[input.seed_index]);

      #pragma unroll
      for (uint32_t i = 0; i < T::loops; i++) {
        float val;
        if constexpr (T::only_a) {
          val = octaves.sample_only_a(shared_grad_dot_table, x, 0, z);
        } else {
          val = octaves.sample(shared_grad_dot_table, x, 0, z);
        }

        const bool valid = (val < T::noise_threshold);
        total_valid += warp_reduce_add((uint32_t)valid);

        if constexpr (T::move_center) {
          if (valid) {
            sum_dx += x - input.x;
            sum_dz += z - input.z;
          }
        }
        z += z_step;
      }
    }

    if constexpr (T::samples > 32) {
      if (is_valid_input && (threadIdx.x % 32 == 0)) {
        atomicAdd(&shared_counts[block_input_index], total_valid);
      }
      __syncthreads();
      if (is_valid_input) {
        total_valid = shared_counts[block_input_index];
      }
    }

    if constexpr (T::move_center) {
      if (is_valid_input) {
        sum_dx = warp_reduce_add(sum_dx);
        sum_dz = warp_reduce_add(sum_dz);
      }
      if constexpr (T::samples > 32) {
        if (is_valid_input && (threadIdx.x % 32 == 0)) {
          atomicAdd(&shared_sums[block_input_index][0], sum_dx);
          atomicAdd(&shared_sums[block_input_index][1], sum_dz);
        }
        __syncthreads();
        if (is_valid_input) {
          sum_dx = shared_sums[block_input_index][0];
          sum_dz = shared_sums[block_input_index][1];
        }
      }
      if (is_valid_input && total_valid != 0) {
        sum_dx /= (int32_t)total_valid;
        sum_dz /= (int32_t)total_valid;
      }
    }

    if (is_valid_input && (total_valid >= T::min_count)) {
      if (pos_index == 0) {
        uint32_t result_index = atomicAdd(outputs.len, 1);
        if (result_index < outputs.max_len) {
          outputs.data[result_index] = {input.seed_index, input.x + sum_dx, input.z + sum_dz};
        }
      }
    }

    if constexpr (T::samples > 32) {
      __syncthreads();
    }
  }
}

template <int32_t NoiseThreshold, size_t Octaves, uint32_t PosRange, uint32_t Samples, uint32_t MinCount, bool FlippedSparseSamples, bool MoveCenter, bool OnlyA>
void Template<NoiseThreshold, Octaves, PosRange, Samples, MinCount, FlippedSparseSamples, MoveCenter, OnlyA>::run(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs, KernelSeed1::Result *results, cudaStream_t stream) {
  using T = Template<NoiseThreshold, Octaves, PosRange, Samples, MinCount, FlippedSparseSamples, MoveCenter, OnlyA>;
  // These stages see very few inputs per launch (filter_2d averages well under
  // one), and at 255 registers only one block fits per SM, so a 8192-block grid
  // costs hundreds of near-empty waves of kernel prologue. The kernel already
  // grid-strides over its inputs, so a smaller grid is equally correct.
  kernel<T><<<256, T::threads_per_block, 0, stream>>>(inputs, outputs, results);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilter2

// cactus was here :)
namespace KernelFilter2_0A {
using T = KernelFilter2::Template<-5500, 3, 8 * 1024, 256, 27, false, false, true>;

static_assert(T::samples == 256);
static_assert(T::samples_square_size == 16);
static_assert(!T::samples_square_sparse);
static_assert(T::octaves == 3 && T::only_a);
static_assert(!T::move_center);

constexpr uint32_t threads_per_block = 256;
constexpr uint32_t warps_per_block = threads_per_block / 32;

constexpr OctaveConfig cfg0 = chosen_continentalness_config.octaves_a[0];
constexpr OctaveConfig cfg1 = chosen_continentalness_config.octaves_a[1];
constexpr float if0 = (float)cfg0.input_factor;
constexpr float if1 = (float)cfg1.input_factor;
constexpr float vf0 = (float)cfg0.value_factor;
constexpr float vf1 = (float)cfg1.value_factor;

__device__ inline void compute_cell(const ImprovedNoise &noise, int32_t int_x, int32_t int_y, int32_t int_z, uint8_t &c000, uint8_t &c100, uint8_t &c010, uint8_t &c110, uint8_t &c001, uint8_t &c101, uint8_t &c011, uint8_t &c111) {
  uint8_t p0 = noise.p[(int_x) & 0xFF];
  uint8_t p1 = noise.p[(int_x + 1) & 0xFF];
  uint8_t p00 = noise.p[(p0 + int_y) & 0xFF];
  uint8_t p01 = noise.p[(p0 + int_y + 1) & 0xFF];
  uint8_t p10 = noise.p[(p1 + int_y) & 0xFF];
  uint8_t p11 = noise.p[(p1 + int_y + 1) & 0xFF];
  c000 = noise.p[(p00 + int_z) & 0xFF];
  c100 = noise.p[(p10 + int_z) & 0xFF];
  c010 = noise.p[(p01 + int_z) & 0xFF];
  c110 = noise.p[(p11 + int_z) & 0xFF];
  c001 = noise.p[(p00 + int_z + 1) & 0xFF];
  c101 = noise.p[(p10 + int_z + 1) & 0xFF];
  c011 = noise.p[(p01 + int_z + 1) & 0xFF];
  c111 = noise.p[(p11 + int_z + 1) & 0xFF];
}

// Perlin's 8-corner gradient dot is affine in dx, and lerp3 nests the fx lerp
// innermost, so for one cell the whole interpolation collapses to
//     result = pa + fx * (qa - pa),  pa = frac_x * ga + ba,  qa = frac_x * gb + bb
// where ga/ba/gb/bb depend only on the cell's eight hashes and the y/z fractions.
// Those are loop-invariant for every sample landing in the same x cell, so a
// sample costs 4 FMAs and no shared-memory reads instead of 24 table reads and
// ~38 flops. Algebraically identical to gradDot + lerp3; the reassociation does
// change float rounding in the last bits, so results are not bit-identical.
// One x-plane of the cell: the four corners sharing a value of i, reduced to the
// pair of coefficients the sample formula needs. Splitting the cell this way lets
// a cell that advances by one in x reuse its predecessor's right face as its own
// left face, which is the common case here -- consecutive samples step a quarter
// of a lattice cell.
__device__ inline void face_coeffs(const GradDotTable &table, const ImprovedNoise &noise,
                                   uint8_t plane, int32_t int_y, int32_t int_z,
                                   float frac_y, float frac_z, float fy, float fz,
                                   float &g, float &b) {
  const uint8_t pj0 = noise.p[(plane + int_y) & 0xFF];
  const uint8_t pj1 = noise.p[(plane + int_y + 1) & 0xFF];
  const uint8_t c00 = noise.p[(pj0 + int_z) & 0xFF];
  const uint8_t c10 = noise.p[(pj1 + int_z) & 0xFF];
  const uint8_t c01 = noise.p[(pj0 + int_z + 1) & 0xFF];
  const uint8_t c11 = noise.p[(pj1 + int_z + 1) & 0xFF];

  const float dy0 = frac_y;
  const float dy1 = frac_y - 1.0f;
  const float dz0 = frac_z;
  const float dz1 = frac_z - 1.0f;

  const uint32_t h00 = c00 & 0xF, h10 = c10 & 0xF, h01 = c01 & 0xF, h11 = c11 & 0xF;
  const float b00 = fmaf(dy0, table.y[h00], dz0 * table.z[h00]);
  const float b10 = fmaf(dy1, table.y[h10], dz0 * table.z[h10]);
  const float b01 = fmaf(dy0, table.y[h01], dz1 * table.z[h01]);
  const float b11 = fmaf(dy1, table.y[h11], dz1 * table.z[h11]);

  // lerp2's first weight walks j (fy), its second walks k (fz).
  g = lerp2(fy, fz, table.x[h00], table.x[h10], table.x[h01], table.x[h11]);
  b = lerp2(fy, fz, b00, b10, b01, b11);
}

__device__ inline void cell_coeffs(const GradDotTable &table, const ImprovedNoise &noise,
                                   int32_t int_x, int32_t int_y, int32_t int_z,
                                   float frac_y, float frac_z, float fy, float fz,
                                   float &ga, float &ba, float &gb, float &bb) {
  face_coeffs(table, noise, noise.p[(int_x) & 0xFF], int_y, int_z, frac_y, frac_z, fy, fz, ga, ba);
  face_coeffs(table, noise, noise.p[(int_x + 1) & 0xFF], int_y, int_z, frac_y, frac_z, fy, fz, gb, bb);
}

__forceinline__ __device__ float cell_sample(float frac_x, float fx, float ga, float ba, float gb, float bb) {
  const float pa = fmaf(frac_x, ga, ba);
  const float qa = fmaf(frac_x - 1.0f, gb, bb);
  return fmaf(fx, qa - pa, pa);
}

// The per-sample work here is already hoisted as far as it goes: the y and z
// setup is computed once per candidate, the eight corner indices are cached
// across consecutive x samples so compute_cell's twelve lookups only rerun when
// the lattice cell changes, and the noise structs are staged per warp.  What is
// left is this function -- eight gradient dots and the trilinear blend, about 62
// of the roughly 70 operations a sample costs.
//
// One further step was tried and rejected. gradDot spends a multiply on
// y * table.y[hash] for each of the eight corners, and y is only ever frac_y or
// frac_y - 1, both fixed for the whole candidate because this kernel samples at
// y = 0. Pre-scaling the gradient's y component into two 16-entry shared tables
// per octave turns each corner from two FMAs and a multiply into two FMAs.
// Measured over six round-robin runs it came out 1.8% slower on this stage: it
// trades an FMA for a shared load, and on Maxwell that is not a trade worth
// making. It would also give up exactness in principle -- the original folds
// y * gy inside an FMA and never rounds it alone -- though as it happened every
// stage output was unchanged.
__device__ inline float interp(const GradDotTable &table, float frac_x, float frac_y, float frac_z, float fx, float fy, float fz, uint8_t c000, uint8_t c100, uint8_t c010, uint8_t c110, uint8_t c001, uint8_t c101, uint8_t c011, uint8_t c111) {
  float n000 = gradDot(table, c000, frac_x, frac_y, frac_z);
  float n100 = gradDot(table, c100, frac_x - 1.0f, frac_y, frac_z);
  float n010 = gradDot(table, c010, frac_x, frac_y - 1.0f, frac_z);
  float n110 = gradDot(table, c110, frac_x - 1.0f, frac_y - 1.0f, frac_z);
  float n001 = gradDot(table, c001, frac_x, frac_y, frac_z - 1.0f);
  float n101 = gradDot(table, c101, frac_x - 1.0f, frac_y, frac_z - 1.0f);
  float n011 = gradDot(table, c011, frac_x, frac_y - 1.0f, frac_z - 1.0f);
  float n111 = gradDot(table, c111, frac_x - 1.0f, frac_y - 1.0f, frac_z - 1.0f);
  return lerp3(fx, fy, fz, n000, n100, n010, n110, n001, n101, n011, n111);
}

__global__ __launch_bounds__(threads_per_block) void kernel(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs, KernelSeed1::Result *results) {
  __shared__ GradDotTable shared_grad_dot_table;
  __shared__ ImprovedNoise s_oct0[warps_per_block];
  __shared__ ImprovedNoise s_oct1[warps_per_block];

  for (uint32_t i = threadIdx.x; i < sizeof(shared_grad_dot_table) / sizeof(uint32_t); i += threads_per_block) {
    reinterpret_cast<uint32_t *>(&shared_grad_dot_table)[i] = reinterpret_cast<uint32_t *>(&device_grad_dot_table)[i];
  }
  __syncthreads();

  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp_in_block = threadIdx.x >> 5;
  const uint32_t warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const uint32_t num_warps = (gridDim.x * blockDim.x) >> 5;

  ImprovedNoise &oct0 = s_oct0[warp_in_block];
  ImprovedNoise &oct1 = s_oct1[warp_in_block];

  const uint32_t z_index = lane >> 1;
  const uint32_t x_start = (lane & 1u) * 8u;

  constexpr uint32_t words = sizeof(ImprovedNoise) / sizeof(uint32_t);

  uint32_t inputs_len = *inputs.len;
  for (uint32_t input_index = warp_global; input_index < inputs_len; input_index += num_warps) {
    const SeedPos input = inputs.data[input_index];
    const uint32_t seed_index = input.seed_index;

    {
      const uint32_t *src0 = reinterpret_cast<const uint32_t *>(&results[seed_index].continentalness_0A);
      const uint32_t *src1 = reinterpret_cast<const uint32_t *>(&results[seed_index].continentalness_1A);
      uint32_t *dst0 = reinterpret_cast<uint32_t *>(&oct0);
      uint32_t *dst1 = reinterpret_cast<uint32_t *>(&oct1);
      for (uint32_t i = lane; i < words; i += 32) {
        dst0[i] = src0[i];
        dst1[i] = src1[i];
      }
    }
    __syncwarp();

    const int32_t z_world = input.z + (int32_t)(z_index * T::pos_step) + T::pos_offset;
    const int32_t x_base = input.x + (int32_t)(x_start * T::pos_step) + T::pos_offset;

    const float y0 = oct0.yo;
    const int32_t int_y0 = __float2int_rd(y0);
    const float frac_y0 = y0 - (float)int_y0;
    const float fy0 = smoothstep(frac_y0);
    const float z0c = z_world * if0 + oct0.zo;
    const int32_t int_z0 = __float2int_rd(z0c);
    const float frac_z0 = z0c - (float)int_z0;
    const float fz0 = smoothstep(frac_z0);

    const float y1 = oct1.yo;
    const int32_t int_y1 = __float2int_rd(y1);
    const float frac_y1 = y1 - (float)int_y1;
    const float fy1 = smoothstep(frac_y1);
    const float z1c = z_world * if1 + oct1.zo;
    const int32_t int_z1 = __float2int_rd(z1c);
    const float frac_z1 = z1c - (float)int_z1;
    const float fz1 = smoothstep(frac_z1);

    int32_t cur_ix0 = 0;
    int32_t cur_ix1 = 0;
    bool have0 = false;
    bool have1 = false;
    float a_ga, a_ba, a_gb, a_bb;
    float b_ga, b_ba, b_gb, b_bb;

    uint32_t local_count = 0;
#pragma unroll
    for (uint32_t k = 0; k < 8; k++) {
      const int32_t x_world = x_base + (int32_t)(k * T::pos_step);

      const float x0c = x_world * if0 + oct0.xo;
      const int32_t int_x0 = __float2int_rd(x0c);
      const float frac_x0 = x0c - (float)int_x0;
      if (!have0 || int_x0 != cur_ix0) {
        if (have0 && int_x0 == cur_ix0 + 1) {
          // Stepped one cell along x, so this cell's left face is the face we
          // already built as the previous cell's right face. 0nly the new right
          // face costs anything.
          a_ga = a_gb;
          a_ba = a_bb;
          face_coeffs(shared_grad_dot_table, oct0, oct0.p[(int_x0 + 1) & 0xFF], int_y0, int_z0, frac_y0, frac_z0, fy0, fz0, a_gb, a_bb);
        } else {
          cell_coeffs(shared_grad_dot_table, oct0, int_x0, int_y0, int_z0, frac_y0, frac_z0, fy0, fz0, a_ga, a_ba, a_gb, a_bb);
        }
        cur_ix0 = int_x0;
        have0 = true;
      }
      const float fx0 = smoothstep(frac_x0);
      const float noise0 = cell_sample(frac_x0, fx0, a_ga, a_ba, a_gb, a_bb);

      const float x1c = x_world * if1 + oct1.xo;
      const int32_t int_x1 = __float2int_rd(x1c);
      const float frac_x1 = x1c - (float)int_x1;
      if (!have1 || int_x1 != cur_ix1) {
        if (have1 && int_x1 == cur_ix1 + 1) {
          // Stepped one cell along x, so this cell's left face is the face we
          // already built as the previous cell's right face. 1nly the new right
          // face costs anything.
          b_ga = b_gb;
          b_ba = b_bb;
          face_coeffs(shared_grad_dot_table, oct1, oct1.p[(int_x1 + 1) & 0xFF], int_y1, int_z1, frac_y1, frac_z1, fy1, fz1, b_gb, b_bb);
        } else {
          cell_coeffs(shared_grad_dot_table, oct1, int_x1, int_y1, int_z1, frac_y1, frac_z1, fy1, fz1, b_ga, b_ba, b_gb, b_bb);
        }
        cur_ix1 = int_x1;
        have1 = true;
      }
      const float fx1 = smoothstep(frac_x1);
      const float noise1 = cell_sample(frac_x1, fx1, b_ga, b_ba, b_gb, b_bb);

      float val = 0;
      val += noise0 * vf0;
      val += noise1 * vf1;

      local_count += (val < T::noise_threshold) ? 1u : 0u;
    }

    const uint32_t total = warp_reduce_add(local_count);
    if (lane == 0 && total >= T::min_count) {
      uint32_t result_index = atomicAdd(outputs.len, 1);
      if (result_index < outputs.max_len){
        outputs.data[result_index] = {seed_index, input.x, input.z};
      }
    }
    __syncwarp();
  }
}

void run(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs, KernelSeed1::Result *results, cudaStream_t stream) {
  kernel<<<32 * 256, threads_per_block, 0, stream>>>(inputs, outputs, results);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilter2_0A

namespace KernelFilter2_0B {
using T = KernelFilter2::Template<-5500, 3, 8 * 1024, 256, 20, false, false, false>;

static_assert(T::samples == 256);
static_assert(T::samples_square_size == 16);
static_assert(!T::samples_square_sparse);
static_assert(T::octaves == 3 && !T::only_a);
static_assert(!T::move_center);

constexpr uint32_t threads_per_block = 256;
constexpr uint32_t warps_per_block = threads_per_block / 32;

constexpr OctaveConfig cfg0 = chosen_continentalness_config.octaves_b[0];
constexpr OctaveConfig cfg1 = chosen_continentalness_config.octaves_b[1];
constexpr float if0 = (float)cfg0.input_factor;
constexpr float if1 = (float)cfg1.input_factor;
constexpr float vf0 = (float)cfg0.value_factor;
constexpr float vf1 = (float)cfg1.value_factor;

__device__ inline void compute_cell(const ImprovedNoise &noise, int32_t int_x, int32_t int_y, int32_t int_z, uint8_t &c000, uint8_t &c100, uint8_t &c010, uint8_t &c110, uint8_t &c001, uint8_t &c101, uint8_t &c011, uint8_t &c111) {
  uint8_t p0 = noise.p[(int_x) & 0xFF];
  uint8_t p1 = noise.p[(int_x + 1) & 0xFF];
  uint8_t p00 = noise.p[(p0 + int_y) & 0xFF];
  uint8_t p01 = noise.p[(p0 + int_y + 1) & 0xFF];
  uint8_t p10 = noise.p[(p1 + int_y) & 0xFF];
  uint8_t p11 = noise.p[(p1 + int_y + 1) & 0xFF];
  c000 = noise.p[(p00 + int_z) & 0xFF];
  c100 = noise.p[(p10 + int_z) & 0xFF];
  c010 = noise.p[(p01 + int_z) & 0xFF];
  c110 = noise.p[(p11 + int_z) & 0xFF];
  c001 = noise.p[(p00 + int_z + 1) & 0xFF];
  c101 = noise.p[(p10 + int_z + 1) & 0xFF];
  c011 = noise.p[(p01 + int_z + 1) & 0xFF];
  c111 = noise.p[(p11 + int_z + 1) & 0xFF];
}

// One x-plane of the cell: the four corners sharing a value of i, reduced to the
// pair of coefficients the sample formula needs. Splitting the cell this way lets
// a cell that advances by one in x reuse its predecessor's right face as its own
// left face, which is the common case here -- consecutive samples step a quarter
// of a lattice cell.
__device__ inline void face_coeffs(const GradDotTable &table, const ImprovedNoise &noise,
                                   uint8_t plane, int32_t int_y, int32_t int_z,
                                   float frac_y, float frac_z, float fy, float fz,
                                   float &g, float &b) {
  const uint8_t pj0 = noise.p[(plane + int_y) & 0xFF];
  const uint8_t pj1 = noise.p[(plane + int_y + 1) & 0xFF];
  const uint8_t c00 = noise.p[(pj0 + int_z) & 0xFF];
  const uint8_t c10 = noise.p[(pj1 + int_z) & 0xFF];
  const uint8_t c01 = noise.p[(pj0 + int_z + 1) & 0xFF];
  const uint8_t c11 = noise.p[(pj1 + int_z + 1) & 0xFF];

  const float dy0 = frac_y;
  const float dy1 = frac_y - 1.0f;
  const float dz0 = frac_z;
  const float dz1 = frac_z - 1.0f;

  const uint32_t h00 = c00 & 0xF, h10 = c10 & 0xF, h01 = c01 & 0xF, h11 = c11 & 0xF;
  const float b00 = fmaf(dy0, table.y[h00], dz0 * table.z[h00]);
  const float b10 = fmaf(dy1, table.y[h10], dz0 * table.z[h10]);
  const float b01 = fmaf(dy0, table.y[h01], dz1 * table.z[h01]);
  const float b11 = fmaf(dy1, table.y[h11], dz1 * table.z[h11]);

  // lerp2's first weight walks j (fy), its second walks k (fz).
  g = lerp2(fy, fz, table.x[h00], table.x[h10], table.x[h01], table.x[h11]);
  b = lerp2(fy, fz, b00, b10, b01, b11);
}

__device__ inline void cell_coeffs(const GradDotTable &table, const ImprovedNoise &noise,
                                   int32_t int_x, int32_t int_y, int32_t int_z,
                                   float frac_y, float frac_z, float fy, float fz,
                                   float &ga, float &ba, float &gb, float &bb) {
  face_coeffs(table, noise, noise.p[(int_x) & 0xFF], int_y, int_z, frac_y, frac_z, fy, fz, ga, ba);
  face_coeffs(table, noise, noise.p[(int_x + 1) & 0xFF], int_y, int_z, frac_y, frac_z, fy, fz, gb, bb);
}

__forceinline__ __device__ float cell_sample(float frac_x, float fx, float ga, float ba, float gb, float bb) {
  const float pa = fmaf(frac_x, ga, ba);
  const float qa = fmaf(frac_x - 1.0f, gb, bb);
  return fmaf(fx, qa - pa, pa);
}

__device__ inline float interp(const GradDotTable &table, float frac_x, float frac_y, float frac_z, float fx, float fy, float fz, uint8_t c000, uint8_t c100, uint8_t c010, uint8_t c110, uint8_t c001, uint8_t c101, uint8_t c011, uint8_t c111) {
  float n000 = gradDot(table, c000, frac_x, frac_y, frac_z);
  float n100 = gradDot(table, c100, frac_x - 1.0f, frac_y, frac_z);
  float n010 = gradDot(table, c010, frac_x, frac_y - 1.0f, frac_z);
  float n110 = gradDot(table, c110, frac_x - 1.0f, frac_y - 1.0f, frac_z);
  float n001 = gradDot(table, c001, frac_x, frac_y, frac_z - 1.0f);
  float n101 = gradDot(table, c101, frac_x - 1.0f, frac_y, frac_z - 1.0f);
  float n011 = gradDot(table, c011, frac_x, frac_y - 1.0f, frac_z - 1.0f);
  float n111 = gradDot(table, c111, frac_x - 1.0f, frac_y - 1.0f, frac_z - 1.0f);
  return lerp3(fx, fy, fz, n000, n100, n010, n110, n001, n101, n011, n111);
}

__global__ __launch_bounds__(threads_per_block) void kernel(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs, KernelSeed1::Result *results) {
  __shared__ GradDotTable shared_grad_dot_table;
  __shared__ ImprovedNoise s_oct0[warps_per_block];
  __shared__ ImprovedNoise s_oct1[warps_per_block];

  for (uint32_t i = threadIdx.x; i < sizeof(shared_grad_dot_table) / sizeof(uint32_t); i += threads_per_block) {
    reinterpret_cast<uint32_t *>(&shared_grad_dot_table)[i] = reinterpret_cast<uint32_t *>(&device_grad_dot_table)[i];
  }
  __syncthreads();

  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp_in_block = threadIdx.x >> 5;
  const uint32_t warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const uint32_t num_warps = (gridDim.x * blockDim.x) >> 5;

  ImprovedNoise &oct0 = s_oct0[warp_in_block];
  ImprovedNoise &oct1 = s_oct1[warp_in_block];

  const uint32_t z_index = lane >> 1;
  const uint32_t x_start = (lane & 1u) * 8u;

  constexpr uint32_t words = sizeof(ImprovedNoise) / sizeof(uint32_t);

  uint32_t inputs_len = *inputs.len;
  for (uint32_t input_index = warp_global; input_index < inputs_len; input_index += num_warps) {
    const SeedPos input = inputs.data[input_index];
    const uint32_t seed_index = input.seed_index;

    {
      const uint32_t *src0 = reinterpret_cast<const uint32_t *>(&results[seed_index].continentalness_0B);
      const uint32_t *src1 = reinterpret_cast<const uint32_t *>(&results[seed_index].continentalness_1B);
      uint32_t *dst0 = reinterpret_cast<uint32_t *>(&oct0);
      uint32_t *dst1 = reinterpret_cast<uint32_t *>(&oct1);
      for (uint32_t i = lane; i < words; i += 32) {
        dst0[i] = src0[i];
        dst1[i] = src1[i];
      }
    }
    __syncwarp();

    const int32_t z_world = input.z + (int32_t)(z_index * T::pos_step) + T::pos_offset;
    const int32_t x_base = input.x + (int32_t)(x_start * T::pos_step) + T::pos_offset;

    const float y0 = oct0.yo;
    const int32_t int_y0 = __float2int_rd(y0);
    const float frac_y0 = y0 - (float)int_y0;
    const float fy0 = smoothstep(frac_y0);
    const float z0c = z_world * if0 + oct0.zo;
    const int32_t int_z0 = __float2int_rd(z0c);
    const float frac_z0 = z0c - (float)int_z0;
    const float fz0 = smoothstep(frac_z0);
    const float y1 = oct1.yo;
    const int32_t int_y1 = __float2int_rd(y1);
    const float frac_y1 = y1 - (float)int_y1;
    const float fy1 = smoothstep(frac_y1);
    const float z1c = z_world * if1 + oct1.zo;
    const int32_t int_z1 = __float2int_rd(z1c);
    const float frac_z1 = z1c - (float)int_z1;
    const float fz1 = smoothstep(frac_z1);

    int32_t cur_ix0 = 0;
    int32_t cur_ix1 = 0;
    bool have0 = false;
    bool have1 = false;
    float a_ga, a_ba, a_gb, a_bb;
    float b_ga, b_ba, b_gb, b_bb;

    uint32_t local_count = 0;
#pragma unroll
    for (uint32_t k = 0; k < 8; k++) {
      const int32_t x_world = x_base + (int32_t)(k * T::pos_step);

      const float x0c = x_world * if0 + oct0.xo;
      const int32_t int_x0 = __float2int_rd(x0c);
      const float frac_x0 = x0c - (float)int_x0;
      if (!have0 || int_x0 != cur_ix0) {
        if (have0 && int_x0 == cur_ix0 + 1) {
          // Stepped one cell along x, so this cell's left face is the face we
          // already built as the previous cell's right face.
          a_ga = a_gb;
          a_ba = a_bb;
          face_coeffs(shared_grad_dot_table, oct0, oct0.p[(int_x0 + 1) & 0xFF], int_y0, int_z0, frac_y0, frac_z0, fy0, fz0, a_gb, a_bb);
        } else {
          cell_coeffs(shared_grad_dot_table, oct0, int_x0, int_y0, int_z0, frac_y0, frac_z0, fy0, fz0, a_ga, a_ba, a_gb, a_bb);
        }
        cur_ix0 = int_x0;
        have0 = true;
      }
      const float fx0 = smoothstep(frac_x0);
      const float noise0 = cell_sample(frac_x0, fx0, a_ga, a_ba, a_gb, a_bb);

      const float x1c = x_world * if1 + oct1.xo;
      const int32_t int_x1 = __float2int_rd(x1c);
      const float frac_x1 = x1c - (float)int_x1;
      if (!have1 || int_x1 != cur_ix1) {
        if (have1 && int_x1 == cur_ix1 + 1) {
          // Stepped one cell along x, so this cell's left face is the face we
          // already built as the previous cell's right face.
          b_ga = b_gb;
          b_ba = b_bb;
          face_coeffs(shared_grad_dot_table, oct1, oct1.p[(int_x1 + 1) & 0xFF], int_y1, int_z1, frac_y1, frac_z1, fy1, fz1, b_gb, b_bb);
        } else {
          cell_coeffs(shared_grad_dot_table, oct1, int_x1, int_y1, int_z1, frac_y1, frac_z1, fy1, fz1, b_ga, b_ba, b_gb, b_bb);
        }
        cur_ix1 = int_x1;
        have1 = true;
      }
      const float fx1 = smoothstep(frac_x1);
      const float noise1 = cell_sample(frac_x1, fx1, b_ga, b_ba, b_gb, b_bb);

      float val = 0;
      val += noise0 * vf0;
      val += noise1 * vf1;

      local_count += (val < T::noise_threshold) ? 1u : 0u;
    }

    const uint32_t total = warp_reduce_add(local_count);
    if (lane == 0 && total >= T::min_count) {
      uint32_t result_index = atomicAdd(outputs.len, 1);
      if (result_index < outputs.max_len){
        outputs.data[result_index] = {seed_index, input.x, input.z};
      }
    }
    __syncwarp();
  }
}

void run(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs, KernelSeed1::Result *results, cudaStream_t stream) {
  kernel<<<32 * 256, threads_per_block, 0, stream>>>(inputs, outputs, results);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilter2_0B

struct CudaEventWrapper {
  cudaEvent_t event;

  CudaEventWrapper() : event(nullptr) { TRY_CUDA(cudaEventCreate(&event)); }

  CudaEventWrapper(CudaEventWrapper &&other) : event(other.event) {
    other.event = nullptr;
  }

  ~CudaEventWrapper() {
    if (event == nullptr)
      return;
    TRY_CUDA(cudaEventDestroy(event));
  }

  void record(cudaStream_t stream = 0) const {
    TRY_CUDA(cudaEventRecord(event, stream));
  }

  float elapsed(const CudaEventWrapper &end) const {
    float ms;
    TRY_CUDA(cudaEventElapsedTime(&ms, event, end.event));
    return ms;
  }

  void synchronize() const { TRY_CUDA(cudaEventSynchronize(event)); }
};

struct StageStats {
  std::string name;
  uint32_t *inputs_len;
  uint32_t *outputs_len;
  uint64_t inputs_multiplier;
  uint32_t max_outputs_len;
  CudaEventWrapper event;
  double total_time;
  uint64_t total_inputs;
  uint64_t total_outputs;

  StageStats(std::string name, uint32_t *inputs_len, uint32_t *outputs_len,
             uint64_t inputs_len_multiplier, uint32_t max_outputs_len)
      : name(std::move(name)), inputs_len(inputs_len), outputs_len(outputs_len),
        inputs_multiplier(inputs_len_multiplier),
        max_outputs_len(max_outputs_len), event(), total_time(), total_inputs(),
        total_outputs() {}

  StageStats(StageStats &&other)
      : name(std::move(other.name)), inputs_len(other.inputs_len),
        outputs_len(other.outputs_len),
        inputs_multiplier(other.inputs_multiplier),
        max_outputs_len(other.max_outputs_len), event(std::move(other.event)),
        total_time(other.total_time), total_inputs(other.total_inputs),
        total_outputs(other.total_outputs) {}

  void record(cudaStream_t stream = 0) { event.record(stream); }

  void update(CudaEventWrapper &prev_event) {
    total_time += prev_event.elapsed(event) * 1e-3;
    total_inputs += inputs_len ? *inputs_len : 1;
    total_outputs += *outputs_len;
    if (*outputs_len > max_outputs_len) {
      std::printf("%s outputs overflow: len = %" PRIu32 " max_len = %" PRIu32
                  "\n",
                  name.c_str(), *outputs_len, max_outputs_len);
    }
  }

  void reset() {
    total_time = 0;
    total_inputs = 0;
    total_outputs = 0;
  }
};

std::pair<double, char> scale_si(double val) {
  std::pair<double, char> units[] = {
      {1e12, 'T'},  
      {1e9, 'G'},
      {1e6, 'M'},
      {1e3, 'k'},
  };
  for (auto [unit_scale, unit] : units) {
    if (val >= unit_scale) {
      return {val / unit_scale, unit};
    }
  }
  return {val, ' '};
}

struct BufferLens {
  uint32_t results_len_filter_seeds;
  uint32_t results_len_filter_gradvecs_1;
  uint32_t results_len_filter_2_0a;
  uint32_t results_len_filter_2_0b;
  uint32_t results_len_filter_gradvecs_2;
  uint32_t results_len_filter_2[7];
};


GpuThread::GpuThread(int device, SeedIterator &input, GpuOutputs &outputs, bool benchmark, std::atomic_bool &running)
    : Thread(), device(device), input(input), outputs(outputs), benchmark(benchmark), running(running) {
  start();
}

void GpuThread::run() {
  std::printf("Initializing device %d\n", device);

  TRY_CUDA(cudaSetDevice(device));
  TRY_CUDA(cudaFuncSetAttribute(KernelFilterGradVecs1::kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  init_grad_dot_table();
  init_conv_kernels();

  cudaStream_t stream;
  TRY_CUDA(cudaStreamCreate(&stream));

  BufferLens host_buffer_lens;
  BufferLens *device_buffer_lens;
  TRY_CUDA(cudaMalloc(&device_buffer_lens, sizeof(*device_buffer_lens)));

  std::printf("Running device %d\n", device);

  DeviceBuffer buffer_seeds(sizeof(uint64_t) * KernelSeed1::threads_per_run);
  DeviceBuffer buffer_results(sizeof(KernelSeed1::Result) * KernelSeed1::threads_per_run);
  DeviceBuffer buffer_late_init_flags(sizeof(uint32_t) * KernelSeed1::threads_per_run);
  DeviceBuffer buffer_1(UINT32_C(1) << 31);
  DeviceBuffer buffer_2(UINT32_C(1) << 29);
  std::vector<SeedPos> h_buffer;
  std::vector<StageStats> stage_stats;
  stage_stats.reserve(32);

  KernelSeed1::Result *results = (KernelSeed1::Result *)buffer_results.data;

  CudaEventWrapper event_start;

  OutputBuffer<uint64_t> outputs_filter_seeds(buffer_seeds, &device_buffer_lens->results_len_filter_seeds);
  auto &stage_filter_seeds = stage_stats.emplace_back("filter_seeds", nullptr, &host_buffer_lens.results_len_filter_seeds, KernelFilterSeeds::threads_per_run, outputs_filter_seeds.max_len);

  auto &stage_init_seeds = stage_stats.emplace_back("init_seeds", stage_filter_seeds.outputs_len, stage_filter_seeds.outputs_len, 1, KernelSeed1::threads_per_run);

  OutputBuffer<SeedPos> outputs_filter_gradvecs_1(buffer_2, &device_buffer_lens->results_len_filter_gradvecs_1);
  auto &stage_filter_gradvecs_1 = stage_stats.emplace_back("filter_gradvecs_1", stage_filter_seeds.outputs_len, &host_buffer_lens.results_len_filter_gradvecs_1, 256 * 256, outputs_filter_gradvecs_1.max_len);

  OutputBuffer<SeedPos> outputs_filter_2_0a(buffer_1, &device_buffer_lens->results_len_filter_2_0a);
  auto &stage_filter_2_0a = stage_stats.emplace_back("filter_2_01a", stage_filter_gradvecs_1.outputs_len, &host_buffer_lens.results_len_filter_2_0a, 1, outputs_filter_2_0a.max_len);

  OutputBuffer<SeedPos> outputs_filter_gradvecs_2(buffer_2, &device_buffer_lens->results_len_filter_gradvecs_2);
  auto &stage_init_seeds_0b = stage_stats.emplace_back("init_seeds_0b", stage_filter_2_0a.outputs_len, stage_filter_2_0a.outputs_len, 1, outputs_filter_2_0a.max_len);
  auto &stage_filter_gradvecs_2 = stage_stats.emplace_back("filter_gradvecs_2", stage_filter_2_0a.outputs_len, &host_buffer_lens.results_len_filter_gradvecs_2, KernelFilterGradVecs2::threads_per_seed, outputs_filter_gradvecs_2.max_len);

  OutputBuffer<SeedPos> outputs_filter_2_0b(buffer_2, &device_buffer_lens->results_len_filter_2_0b);
  auto &stage_filter_2_0b = stage_stats.emplace_back("filter_2_01b", stage_filter_gradvecs_2.outputs_len, &host_buffer_lens.results_len_filter_2_0b, 1, outputs_filter_2_0b.max_len);

  auto &stage_init_seeds_late_1 = stage_stats.emplace_back("init_seeds_late_1", stage_filter_2_0b.outputs_len, stage_filter_2_0b.outputs_len, 1, outputs_filter_2_0b.max_len);

  using Kernel2RunFunc = void (*)(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs, KernelSeed1::Result *results, cudaStream_t stream);
  struct Filter2Stage {
    Kernel2RunFunc run;
    OutputBuffer<SeedPos> outputs;
    StageStats &stage;
    StageStats *late_stage;

    Filter2Stage(Kernel2RunFunc run, OutputBuffer<SeedPos> outputs, StageStats &stage, StageStats *late_stage)
        : run(run), outputs(outputs), stage(stage), late_stage(late_stage) {}
  };
  
  Kernel2RunFunc filter_2_0A_run = KernelFilter2_0A::run;
  Kernel2RunFunc filter_2_0B_run = KernelFilter2_0B::run;

  Kernel2RunFunc filter_2_runs[] = {
      KernelFilter2::Template<-10500, 12, 8 * 1024, 256, 24, false, true, false>::run, 
      KernelFilter2::Template<-10500, 14, 8 * 1024, 1024, 110, false, true, false>::run,
      KernelFilter2::Template<-10500, 16, 10 * 1024, 4096, 340, false, false, false>::run,
      KernelFilter2::Template<-10500, 18, 10 * 1024, 16384, 1540, false, false, false>::run, // zajonc was here :D, use 1600 instead of 1540 because of colab shitty cpus
  };
  std::vector<Filter2Stage> filter_2;
  {
    uint32_t *inputs_len = stage_filter_2_0b.outputs_len;
    for (size_t i = 0; i < sizeof(filter_2_runs) / sizeof(*filter_2_runs); i++) {
      OutputBuffer<SeedPos> outputs(i % 2 == 0 ? buffer_1 : buffer_2, &device_buffer_lens->results_len_filter_2[i]);

      uint32_t *outputs_len = &host_buffer_lens.results_len_filter_2[i];
      auto &stage = stage_stats.emplace_back(std::string("filter_2") + (char)('a' + i), inputs_len, outputs_len, 1, outputs.max_len);
      StageStats *late_stage = nullptr;
      if (i < 3) {
        late_stage = &stage_stats.emplace_back(std::string("init_seeds_late_") + (char)('2' + i), outputs_len, outputs_len, 1, outputs.max_len);
      }
      inputs_len = outputs_len;

      filter_2.emplace_back(filter_2_runs[i], outputs, stage, late_stage);
    }
  }

  auto start = std::chrono::steady_clock::now();

  for (uint32_t i = 0; !should_stop() && (!benchmark || i < PRINT_INTERVAL); i++) {
    uint64_t start_seed = input.next(KernelFilterSeeds::threads_per_run);

    TRY_CUDA(cudaMemsetAsync(device_buffer_lens, 0, sizeof(*device_buffer_lens), stream));

    event_start.record(stream);

    KernelFilterSeeds::run(start_seed, outputs_filter_seeds, stream);
    stage_filter_seeds.record(stream);

    KernelSeed1::kernel<<<KernelSeed1::threads_per_run / KernelSeed1::threads_per_block, KernelSeed1::threads_per_block, 0, stream>>>(outputs_filter_seeds, results);
    TRY_CUDA(cudaGetLastError());
    stage_init_seeds.record(stream);

    KernelFilterGradVecs1::run(outputs_filter_seeds, outputs_filter_gradvecs_1, results, stream);
    stage_filter_gradvecs_1.record(stream);

    filter_2_0A_run(outputs_filter_gradvecs_1, outputs_filter_2_0a, results, stream);
    stage_filter_2_0a.record(stream);

    TRY_CUDA(cudaMemsetAsync(buffer_late_init_flags.data, 0, sizeof(uint32_t) * KernelSeed1::threads_per_run, stream));
    KernelSeed1::run_late<1>(outputs_filter_seeds, outputs_filter_2_0a, results, (uint32_t *)buffer_late_init_flags.data, stream);
    stage_init_seeds_0b.record(stream);

    KernelFilterGradVecs2::run(outputs_filter_2_0a, outputs_filter_gradvecs_2, results, stream);
    stage_filter_gradvecs_2.record(stream);

    filter_2_0B_run(outputs_filter_gradvecs_2, outputs_filter_2_0b, results, stream);
    stage_filter_2_0b.record(stream);

    KernelSeed1::run_late<2>(outputs_filter_seeds, outputs_filter_2_0b, results, (uint32_t *)buffer_late_init_flags.data, stream);
    stage_init_seeds_late_1.record(stream);

    {
      OutputBuffer<SeedPos> *inputs = &outputs_filter_2_0b;
      for (size_t filter_index = 0; filter_index < filter_2.size(); filter_index++) {
        auto &filter = filter_2[filter_index];

        filter.run(*inputs, filter.outputs, results, stream);
        filter.stage.record(stream);

        if (filter_index == 0) {
          KernelSeed1::run_late<3>(outputs_filter_seeds, filter.outputs, results, (uint32_t *)buffer_late_init_flags.data, stream);
        } else if (filter_index == 1) {
          KernelSeed1::run_late<4>(outputs_filter_seeds, filter.outputs, results, (uint32_t *)buffer_late_init_flags.data, stream);
        } else if (filter_index == 2) {
          KernelSeed1::run_late<5>(outputs_filter_seeds, filter.outputs, results, (uint32_t *)buffer_late_init_flags.data, stream);
        }

        if (filter.late_stage != nullptr) {
          filter.late_stage->record(stream);
        }

        inputs = &filter.outputs;
      }
    }

    TRY_CUDA(cudaMemcpyAsync(&host_buffer_lens, device_buffer_lens, sizeof(host_buffer_lens), cudaMemcpyDeviceToHost, stream));

    TRY_CUDA(cudaStreamSynchronize(stream));

#ifdef DIAG_GRADVECS_1
    {
      const uint32_t g1 = host_buffer_lens.results_len_filter_gradvecs_1;
      long long xmin = (1ll << 62), xmax = -(1ll << 62), zmin = (1ll << 62), zmax = -(1ll << 62);
      uint32_t g1tgt = 0;
      std::vector<SeedPos> diag_buf(std::max(1u, g1));
      if (g1) {
        TRY_CUDA(cudaMemcpy(diag_buf.data(), outputs_filter_gradvecs_1.data, sizeof(SeedPos) * g1, cudaMemcpyDeviceToHost));
        for (uint32_t q = 0; q < g1; q++) {
          if (diag_buf[q].seed_index != 0) continue;
          g1tgt++;
          xmin = std::min<long long>(xmin, diag_buf[q].x); xmax = std::max<long long>(xmax, diag_buf[q].x);
          zmin = std::min<long long>(zmin, diag_buf[q].z); zmax = std::max<long long>(zmax, diag_buf[q].z);
        }
      }
      const uint32_t a1 = host_buffer_lens.results_len_filter_2_0a;
      long long axmin = (1ll << 62), axmax = -(1ll << 62), azmin = (1ll << 62), azmax = -(1ll << 62);
      uint32_t a1tgt = 0;
      std::vector<SeedPos> diag_buf2(std::max(1u, a1));
      if (a1) {
        TRY_CUDA(cudaMemcpy(diag_buf2.data(), outputs_filter_2_0a.data, sizeof(SeedPos) * a1, cudaMemcpyDeviceToHost));
        for (uint32_t q = 0; q < a1; q++) {
          if (diag_buf2[q].seed_index != 0) continue;
          a1tgt++;
          axmin = std::min<long long>(axmin, diag_buf2[q].x); axmax = std::max<long long>(axmax, diag_buf2[q].x);
          azmin = std::min<long long>(azmin, diag_buf2[q].z); azmax = std::max<long long>(azmax, diag_buf2[q].z);
        }
      }
      std::printf("DIAG_SEED start_seed=%" PRIi64 " g1tgt=%u g1_x[%lld,%lld] g1_z[%lld,%lld] a1tgt=%u a1_x[%lld,%lld] a1_z[%lld,%lld]\n",
        start_seed, g1tgt, xmin, xmax, zmin, zmax, a1tgt, axmin, axmax, azmin, azmax);
      fflush(stdout);
    }
#endif

    host_buffer_lens.results_len_filter_seeds = std::min(host_buffer_lens.results_len_filter_seeds, outputs_filter_seeds.max_len);
    host_buffer_lens.results_len_filter_gradvecs_1 = std::min(host_buffer_lens.results_len_filter_gradvecs_1, outputs_filter_gradvecs_1.max_len);
    host_buffer_lens.results_len_filter_2_0a = std::min(host_buffer_lens.results_len_filter_2_0a, outputs_filter_2_0a.max_len);
    host_buffer_lens.results_len_filter_gradvecs_2 = std::min(host_buffer_lens.results_len_filter_gradvecs_2, outputs_filter_gradvecs_2.max_len);
    host_buffer_lens.results_len_filter_2_0b = std::min(host_buffer_lens.results_len_filter_2_0b, outputs_filter_2_0b.max_len);

    for (size_t k = 0; k < filter_2.size(); k++) {
      host_buffer_lens.results_len_filter_2[k] = std::min(host_buffer_lens.results_len_filter_2[k], filter_2[k].outputs.max_len);
    }

    {
      CudaEventWrapper *prev_event = &event_start;
      for (auto &stage : stage_stats) {
        stage.update(*prev_event);
        prev_event = &stage.event;
      }
    }

    const auto &final_outputs = filter_2.back().outputs;
    const auto &final_outputs_len = *filter_2.back().stage.outputs_len;
    if (final_outputs_len > 0) {
      // uint32_t len = std::min(final_outputs_len, UINT32_C(10));
      uint32_t len = final_outputs_len;
      h_buffer.resize(len);
      TRY_CUDA(cudaMemcpy(h_buffer.data(), final_outputs.data, sizeof(*h_buffer.data()) * len, cudaMemcpyDeviceToHost));

      {
        std::lock_guard lock(outputs.mutex);
        for (const auto &result : h_buffer) {
          uint64_t seed;
          TRY_CUDA(cudaMemcpy(&seed, &outputs_filter_seeds.data[result.seed_index], sizeof(seed), cudaMemcpyDeviceToHost));
          outputs.queue.push({seed, result.x * 4, result.z * 4});
        }
      }
    }

    if ((i + 1) % PRINT_INTERVAL == 0) {
      auto end = std::chrono::steady_clock::now();
      double host_total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;

      std::printf("\n");
      std::printf("start_seed = %" PRIi64 "\n", start_seed);

      uint64_t total_inputs = stage_filter_seeds.total_inputs * stage_filter_seeds.inputs_multiplier;
      uint64_t total_outputs = filter_2.back().stage.total_outputs;

      // Find max input value for dynamic column width
      uint64_t max_inputs = total_inputs;
      uint64_t gradvecs_1_inputs = stage_filter_gradvecs_1.total_inputs * stage_filter_gradvecs_1.inputs_multiplier;
      if (gradvecs_1_inputs > max_inputs) max_inputs = gradvecs_1_inputs;
      int inputs_width = std::max(12, (int)std::snprintf(nullptr, 0, "%" PRIu64, max_inputs));
      int outputs_width = 12;

      // Find max time value for dynamic time column width
      double max_time_ms = host_total_time * 1e3;
      for (auto &stage : stage_stats) {
        double stage_time_ms = stage.total_time * 1e3;
        if (stage_time_ms > max_time_ms) max_time_ms = stage_time_ms;
      }
      int time_width = std::max(9, (int)std::snprintf(nullptr, 0, "%.3f", max_time_ms));

      double kernel_total_time = 0;
      for (auto &stage : stage_stats) {
        uint64_t scaled_total_inputs = stage.total_inputs * stage.inputs_multiplier;
        auto [scaled_input_speed, input_speed_unit] = scale_si(scaled_total_inputs / stage.total_time);
        auto [scaled_output_speed, output_speed_unit] = scale_si(stage.total_outputs / stage.total_time);
        std::printf("%-20s - %*.3f ms | %7.3f %% | %*" PRIu64 " -> %*" PRIu64
                    " | 1 in %11.3f | %7.3f %cips | %7.3f %cops\n",
                    stage.name.c_str(), time_width, stage.total_time * 1e3,
                    stage.total_time / host_total_time * 100.0,
                    inputs_width, scaled_total_inputs, outputs_width, stage.total_outputs,
                    (double)scaled_total_inputs / stage.total_outputs,
                    scaled_input_speed, input_speed_unit, scaled_output_speed,
                    output_speed_unit);
        kernel_total_time += stage.total_time;
      }

      auto [scaled_input_speed, input_speed_unit] = scale_si(total_inputs / host_total_time);
      auto [scaled_output_speed, output_speed_unit] = scale_si(total_outputs / host_total_time);
      std::printf(
          "%-20s - %*.3f ms | %7.3f %% | %*" PRIu64
          " -> %*" PRIu64 " |                  | %7.3f %cips | %7.3f %cops\n",
          "total", time_width, host_total_time * 1e3, kernel_total_time / host_total_time * 100.0,
          inputs_width, total_inputs, outputs_width, total_outputs, scaled_input_speed, input_speed_unit,
          scaled_output_speed, output_speed_unit);

      size_t gpu_outputs_size;
      {
        std::lock_guard lock(outputs.mutex);
        gpu_outputs_size = outputs.queue.size();
      }
      std::printf("gpu_outputs.size() = %zu\n", gpu_outputs_size);

      for (auto &stage_stat : stage_stats) {
        stage_stat.reset();
      }
      start = end;
    }
  }

  if (benchmark) {
    running.store(false, std::memory_order_relaxed);
  }

  TRY_CUDA(cudaStreamDestroy(stream));
  TRY_CUDA(cudaFree(device_buffer_lens));
}
