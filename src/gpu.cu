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
  
  return ((r >> 32) & 0xFFFFFF) * 5.9604645E-8f;
}

__global__ __launch_bounds__(threads_per_block) void kernel(uint64_t start_seed, OutputBuffer<uint64_t> outputs) {
  constexpr float maxScore = 0.038f; // 0.045f, 0.035f, 0.03f, 0.025f  ==  1 in 2700, 9400, 26000, 54000

  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t seed = start_seed + index;

  const auto seed_fork = XrsrRandom_seed_fork(seed);
  auto noise_random = seed_fork.from(device_chosen_continentalness_config.fork_hash);

  const auto noise_a_yo_fork = noise_yo_fork(noise_random.fork());
  
  float c_0A_yo = octave_yo_mod1<chosen_continentalness_config.octaves_a[0]>(noise_a_yo_fork);
  float score = 0.35f * fabsf(c_0A_yo - 0.5f);
  if (score >= maxScore) {
    return;
  }

  const auto noise_b_yo_fork = noise_yo_fork(noise_random.fork());
  float c_0B_yo = octave_yo_mod1<chosen_continentalness_config.octaves_b[0]>(noise_b_yo_fork);
  score += 0.35f * fabsf(c_0B_yo - 0.5f);
  if (score >= maxScore) {
    return;
  }

  float c_1A_yo = octave_yo_mod1<chosen_continentalness_config.octaves_a[1]>(noise_a_yo_fork);
  score += 0.11f * fabsf(c_1A_yo - 0.5f);
  if (score >= maxScore) {
    return;
  }

  float c_1B_yo = octave_yo_mod1<chosen_continentalness_config.octaves_b[1]>(noise_b_yo_fork);
  score += 0.11f * fabsf(c_1B_yo - 0.5f);
  if (score >= maxScore) {
    return;
  }

  float c_2A_yo = octave_yo_mod1<chosen_continentalness_config.octaves_a[2]>(noise_a_yo_fork);
  score += 0.04f * fabsf(c_2A_yo - 0.5f);
  if (score >= maxScore) {
    return;
  }

  float c_2B_yo = octave_yo_mod1<chosen_continentalness_config.octaves_b[2]>(noise_b_yo_fork);
  score += 0.04f * fabsf(c_2B_yo - 0.5f);
  if (score >= maxScore) {
    return;
  }

  uint32_t result_index = atomicAdd(outputs.len, 1);
  if (result_index >= outputs.max_len){
    return;
  }
  outputs.data[result_index] = seed;
}

void run(uint64_t start_seed, OutputBuffer<uint64_t> outputs, cudaStream_t stream) {
  kernel<<<threads_per_run / threads_per_block, threads_per_block, 0, stream>>>(start_seed, outputs);
  TRY_CUDA(cudaGetLastError());
}
} // namespace KernelFilterSeeds

struct SeedPos {
  uint32_t seed_index;
  int32_t x;
  int32_t z;
};

namespace KernelSeed1 {
constexpr uint32_t threads_per_run = UINT64_C(1) << 16;
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

  init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[0].fork_hash, results, &Result::continentalness_0B, block_base, input_len, active);
  init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[1].fork_hash, results, &Result::continentalness_1B, block_base, input_len, active);
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
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[2].fork_hash, results, &Result::continentalness_2A, seed_index);
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[3].fork_hash, results, &Result::continentalness_3A, seed_index);
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[4].fork_hash, results, &Result::continentalness_4A, seed_index);
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[5].fork_hash, results, &Result::continentalness_5A, seed_index);

    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[2].fork_hash, results, &Result::continentalness_2B, seed_index);
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[3].fork_hash, results, &Result::continentalness_3B, seed_index);
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[4].fork_hash, results, &Result::continentalness_4B, seed_index);
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[5].fork_hash, results, &Result::continentalness_5B, seed_index);
    } else if constexpr (Stage == 2) {
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[6].fork_hash, results, &Result::continentalness_6A, seed_index);
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[6].fork_hash, results, &Result::continentalness_6B, seed_index);
    } else if constexpr (Stage == 3) {
    init_octave_direct(noise, noise_a_fork, device_chosen_continentalness_config.octaves_a[7].fork_hash, results, &Result::continentalness_7A, seed_index);
    init_octave_direct(noise, noise_b_fork, device_chosen_continentalness_config.octaves_b[7].fork_hash, results, &Result::continentalness_7B, seed_index);
    } else if constexpr (Stage == 4) {
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

constexpr float kGradVecs1PrefilterThreshold = -12.0f;
constexpr float kGradVecs1FinalThreshold      = -18.5f;

constexpr float kGradVecs2PrefilterThreshold = -13.5f;
constexpr float kGradVecs2FinalThreshold      = -20.0f;

template <typename IndexT>
__device__ __forceinline__ float score_center_2x2(
    const float conv_z0[513][6],
    const float conv_z1[513][6],
    const IndexT* idx0,
    const IndexT* idx1)
{
  return
      conv_z0[idx0[2]][2] +
      conv_z0[idx0[3]][3] +
      conv_z1[idx1[2]][2] +
      conv_z1[idx1[3]][3];
}

template <typename IndexT>
__device__ __forceinline__ float score_full_12(
    const float conv_z0[513][6],
    const float conv_z1[513][6],
    const IndexT* idx0,
    const IndexT* idx1)
{
  float score = 0.0f;
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    score += conv_z0[idx0[i]][i];
    score += conv_z1[idx1[i]][i];
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
  __shared__ alignas(16) float shared_kernel_0A[6][6][16][2];

  __shared__ float conv_z0[513][6];
  __shared__ float conv_z1[513][6];

  __shared__ alignas(16) uint8_t idx_xy[2][272];

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
      for (int32_t dnx = 0; dnx < 6; ++dnx) {
        float conv0 = 0.0f;
        float conv1 = 0.0f;

#pragma unroll
        for (int32_t dnz = 0; dnz < 6; ++dnz) {
          const uint32_t p = p_z[dnz];
          conv0 += shared_kernel_0A[dnx][dnz][p][0];
          conv1 += shared_kernel_0A[dnx][dnz][p][1];
        }

        conv_z0[nz][dnx] = conv0;
        conv_z1[nz][dnx] = conv1;
        conv_z0[nz + 256][dnx] = conv0;
        conv_z1[nz + 256][dnx] = conv1;
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

    __syncthreads();

    int32_t x = x_center;
    const int32_t z = z_center + nz * cell_size;

    uchar4 c0_0 = *reinterpret_cast<const uchar4*>(&idx_xy[0][0]);
    uchar4 c0_1 = *reinterpret_cast<const uchar4*>(&idx_xy[0][4]);
    uchar4 c1_0 = *reinterpret_cast<const uchar4*>(&idx_xy[1][0]);
    uchar4 c1_1 = *reinterpret_cast<const uchar4*>(&idx_xy[1][4]);

    for (int32_t nx = 0; nx < 256; nx += 8) {
      uchar4 c0_2 = *reinterpret_cast<const uchar4*>(&idx_xy[0][nx + 8]);
      uchar4 c0_3 = *reinterpret_cast<const uchar4*>(&idx_xy[0][nx + 12]);
      uchar4 c1_2 = *reinterpret_cast<const uchar4*>(&idx_xy[1][nx + 8]);
      uchar4 c1_3 = *reinterpret_cast<const uchar4*>(&idx_xy[1][nx + 12]);

      uint16_t w0[13];
      w0[0]  = c0_0.x + nz; w0[1]  = c0_0.y + nz; w0[2]  = c0_0.z + nz; w0[3]  = c0_0.w + nz;
      w0[4]  = c0_1.x + nz; w0[5]  = c0_1.y + nz; w0[6]  = c0_1.z + nz; w0[7]  = c0_1.w + nz;
      w0[8]  = c0_2.x + nz; w0[9]  = c0_2.y + nz; w0[10] = c0_2.z + nz; w0[11] = c0_2.w + nz;
      w0[12] = c0_3.x + nz;

      uint16_t w1[13];
      w1[0]  = c1_0.x + nz; w1[1]  = c1_0.y + nz; w1[2]  = c1_0.z + nz; w1[3]  = c1_0.w + nz;
      w1[4]  = c1_1.x + nz; w1[5]  = c1_1.y + nz; w1[6]  = c1_1.z + nz; w1[7]  = c1_1.w + nz;
      w1[8]  = c1_2.x + nz; w1[9]  = c1_2.y + nz; w1[10] = c1_2.z + nz; w1[11] = c1_2.w + nz;
      w1[12] = c1_3.x + nz;

#pragma unroll
      for (int candidate = 0; candidate < 8; ++candidate) {
        const uint16_t* cw0 = &w0[candidate];
        const uint16_t* cw1 = &w1[candidate];

        const float gate = score_center_2x2(conv_z0, conv_z1, cw0, cw1);
        if (gate >= kGradVecs1PrefilterThreshold) {
          const float score = score_full_12(conv_z0, conv_z1, cw0, cw1);
          if (score > kGradVecs1FinalThreshold) {
            uint32_t res_idx = atomicAdd(outputs.len, 1);
            if (res_idx < outputs.max_len) {
              outputs.data[res_idx] = {seed_index, x + candidate * cell_size, z};
            }
          }
        }
      }

      x += 8 * cell_size;

      c0_0 = c0_2;
      c0_1 = c0_3;
      c1_0 = c1_2;
      c1_1 = c1_3;
    }
  }
}

void run(
    const InputBuffer<uint64_t> seeds,
    OutputBuffer<SeedPos> outputs,
    const KernelSeed1::Result* __restrict__ results,
    cudaStream_t stream)
{
  kernel<<<2048, block_dim_x, 0, stream>>>(seeds, outputs, results);
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

  __shared__ float conv_z0[513][6];
  __shared__ float conv_z1[513][6];

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

        conv_z0[V][dnx] = conv0;
        conv_z1[V][dnx] = conv1;
        conv_z0[V + 256][dnx] = conv0;
        conv_z1[V + 256][dnx] = conv1;
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

        const float gate = score_center_2x2(conv_z0, conv_z1, idx0, idx1);
        if (gate >= kGradVecs2PrefilterThreshold) {
          const float score = score_full_12(conv_z0, conv_z1, idx0, idx1);
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
  kernel<T><<<32 * 256, T::threads_per_block, 0, stream>>>(inputs, outputs, results);
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
    uint8_t a000, a100, a010, a110, a001, a101, a011, a111;
    uint8_t b000, b100, b010, b110, b001, b101, b011, b111;

    uint32_t local_count = 0;
#pragma unroll
    for (uint32_t k = 0; k < 8; k++) {
      const int32_t x_world = x_base + (int32_t)(k * T::pos_step);

      const float x0c = x_world * if0 + oct0.xo;
      const int32_t int_x0 = __float2int_rd(x0c);
      const float frac_x0 = x0c - (float)int_x0;
      if (!have0 || int_x0 != cur_ix0) {
        compute_cell(oct0, int_x0, int_y0, int_z0, a000, a100, a010, a110, a001, a101, a011, a111);
        cur_ix0 = int_x0;
        have0 = true;
      }
      const float fx0 = smoothstep(frac_x0);
      const float noise0 = interp(shared_grad_dot_table, frac_x0, frac_y0, frac_z0, fx0, fy0, fz0, a000, a100, a010, a110, a001, a101, a011, a111);

      const float x1c = x_world * if1 + oct1.xo;
      const int32_t int_x1 = __float2int_rd(x1c);
      const float frac_x1 = x1c - (float)int_x1;
      if (!have1 || int_x1 != cur_ix1) {
        compute_cell(oct1, int_x1, int_y1, int_z1, b000, b100, b010, b110, b001, b101, b011, b111);
        cur_ix1 = int_x1;
        have1 = true;
      }
      const float fx1 = smoothstep(frac_x1);
      const float noise1 = interp(shared_grad_dot_table, frac_x1, frac_y1, frac_z1, fx1, fy1, fz1, b000, b100, b010, b110, b001, b101, b011, b111);

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
    uint8_t a000, a100, a010, a110, a001, a101, a011, a111;
    uint8_t b000, b100, b010, b110, b001, b101, b011, b111;

    uint32_t local_count = 0;
#pragma unroll
    for (uint32_t k = 0; k < 8; k++) {
      const int32_t x_world = x_base + (int32_t)(k * T::pos_step);

      const float x0c = x_world * if0 + oct0.xo;
      const int32_t int_x0 = __float2int_rd(x0c);
      const float frac_x0 = x0c - (float)int_x0;
      if (!have0 || int_x0 != cur_ix0) {
        compute_cell(oct0, int_x0, int_y0, int_z0, a000, a100, a010, a110, a001, a101, a011, a111);
        cur_ix0 = int_x0;
        have0 = true;
      }
      const float fx0 = smoothstep(frac_x0);
      const float noise0 = interp(shared_grad_dot_table, frac_x0, frac_y0, frac_z0, fx0, fy0, fz0, a000, a100, a010, a110, a001, a101, a011, a111);

      const float x1c = x_world * if1 + oct1.xo;
      const int32_t int_x1 = __float2int_rd(x1c);
      const float frac_x1 = x1c - (float)int_x1;
      if (!have1 || int_x1 != cur_ix1) {
        compute_cell(oct1, int_x1, int_y1, int_z1, b000, b100, b010, b110, b001, b101, b011, b111);
        cur_ix1 = int_x1;
        have1 = true;
      }
      const float fx1 = smoothstep(frac_x1);
      const float noise1 = interp(shared_grad_dot_table, frac_x1, frac_y1, frac_z1, fx1, fy1, fz1, b000, b100, b010, b110, b001, b101, b011, b111);

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


GpuThread::GpuThread(int device, SeedIterator &input, GpuOutputs &outputs)
    : Thread(), device(device), input(input), outputs(outputs) {
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
  stage_stats.reserve(16);

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

  for (uint32_t i = 0; !should_stop(); i++) {
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

    KernelFilterGradVecs2::run(outputs_filter_2_0a, outputs_filter_gradvecs_2, results, stream);
    stage_filter_gradvecs_2.record(stream);

    filter_2_0B_run(outputs_filter_gradvecs_2, outputs_filter_2_0b, results, stream);
    stage_filter_2_0b.record(stream);

    TRY_CUDA(cudaMemsetAsync(buffer_late_init_flags.data, 0, sizeof(uint32_t) * KernelSeed1::threads_per_run, stream));
    KernelSeed1::run_late<1>(outputs_filter_seeds, outputs_filter_2_0b, results, (uint32_t *)buffer_late_init_flags.data, stream);
    stage_init_seeds_late_1.record(stream);

    {
      OutputBuffer<SeedPos> *inputs = &outputs_filter_2_0b;
      for (size_t filter_index = 0; filter_index < filter_2.size(); filter_index++) {
        auto &filter = filter_2[filter_index];

        filter.run(*inputs, filter.outputs, results, stream);
        filter.stage.record(stream);

        if (filter_index == 0) {
          KernelSeed1::run_late<2>(outputs_filter_seeds, filter.outputs, results, (uint32_t *)buffer_late_init_flags.data, stream);
        } else if (filter_index == 1) {
          KernelSeed1::run_late<3>(outputs_filter_seeds, filter.outputs, results, (uint32_t *)buffer_late_init_flags.data, stream);
        } else if (filter_index == 2) {
          KernelSeed1::run_late<4>(outputs_filter_seeds, filter.outputs, results, (uint32_t *)buffer_late_init_flags.data, stream);
        }

        if (filter.late_stage != nullptr) {
          filter.late_stage->record(stream);
        }

        inputs = &filter.outputs;
      }
    }

    TRY_CUDA(cudaMemcpyAsync(&host_buffer_lens, device_buffer_lens, sizeof(host_buffer_lens), cudaMemcpyDeviceToHost, stream));

    TRY_CUDA(cudaStreamSynchronize(stream));

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

      double kernel_total_time = 0;
      for (auto &stage : stage_stats) {
        uint64_t scaled_total_inputs = stage.total_inputs * stage.inputs_multiplier;
        auto [scaled_input_speed, input_speed_unit] = scale_si(scaled_total_inputs / stage.total_time);
        auto [scaled_output_speed, output_speed_unit] = scale_si(stage.total_outputs / stage.total_time);
        std::printf("%-20s - %9.3f ms | %7.3f %% | %12" PRIu64 " -> %12" PRIu64
                    " | 1 in %11.3f | %7.3f %cips | %7.3f %cops\n",
                    stage.name.c_str(), stage.total_time * 1e3,
                    stage.total_time / host_total_time * 100.0,
                    scaled_total_inputs, stage.total_outputs,
                    (double)scaled_total_inputs / stage.total_outputs,
                    scaled_input_speed, input_speed_unit, scaled_output_speed,
                    output_speed_unit);
        kernel_total_time += stage.total_time;
      }

      uint64_t total_inputs = stage_filter_seeds.total_inputs * stage_filter_seeds.inputs_multiplier;
      uint64_t total_outputs = filter_2.back().stage.total_outputs;
      auto [scaled_input_speed, input_speed_unit] = scale_si(total_inputs / host_total_time);
      auto [scaled_output_speed, output_speed_unit] = scale_si(total_outputs / host_total_time);
      std::printf(
          "total                - %9.3f ms | %7.3f %% | %12" PRIu64
          " -> %12" PRIu64 " |                  | %7.3f %cips | %7.3f %cops\n",
          host_total_time * 1e3, kernel_total_time / host_total_time * 100.0,
          total_inputs, total_outputs, scaled_input_speed, input_speed_unit,
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

  TRY_CUDA(cudaStreamDestroy(stream));
  TRY_CUDA(cudaFree(device_buffer_lens));
}
