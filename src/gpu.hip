#include <hip/hip_runtime.h>
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
#include <vector>
#include <string>
#include <memory>

#define PANIC(...)                                                             \
  {                                                                            \
    std::fprintf(stderr, __VA_ARGS__);                                         \
    std::abort();                                                              \
  }

#define TRY_HIP(expr) try_hip(expr, __FILE__, __LINE__)

#ifndef PRINT_INTERVAL
#define PRINT_INTERVAL 4096
#endif

void try_hip(hipError_t error, const char *file, uint64_t line) {
  if (error == hipSuccess)
    return;

  PANIC("HIP error at %s:%" PRIu64 ": %s (%d)\n", file, line,
        hipGetErrorString(error), error);
}

struct HipEventWrapper {
  hipEvent_t event;

  HipEventWrapper() : event(nullptr) { TRY_HIP(hipEventCreate(&event)); }

  HipEventWrapper(HipEventWrapper &&other) : event(other.event) {
    other.event = nullptr;
  }

  ~HipEventWrapper() {
    if (event == nullptr)
      return;
    TRY_HIP(hipEventDestroy(event));
  }

  void record(hipStream_t stream = 0) const {
    TRY_HIP(hipEventRecord(event, stream));
  }

  float elapsed(const HipEventWrapper &end) const {
    float ms;
    TRY_HIP(hipEventElapsedTime(&ms, event, end.event));
    return ms;
  }

  void synchronize() const { TRY_HIP(hipEventSynchronize(event)); }
};

constexpr bool is_power_of_two(uint32_t val) { return (val & (val - 1)) == 0; }

constexpr uint32_t compile_time_log2(uint32_t val) { return 31 - std::countl_zero(val); }

#if defined(__HIP__) || defined(__HIPCC__)
#define MASK_ALL (~0ULL)
#else
#define MASK_ALL 0xFFFFFFFFu
#endif

template <typename T> __device__ T reduce_add_32(T val) {
#if __HIP_ARCH__ >= 800
  return __reduce_add_sync(MASK_ALL, val);
#else
  val += __shfl_down_sync(MASK_ALL, val, 1, 32);
  val += __shfl_down_sync(MASK_ALL, val, 2, 32);
  val += __shfl_down_sync(MASK_ALL, val, 4, 32);
  val += __shfl_down_sync(MASK_ALL, val, 8, 32);
  val += __shfl_down_sync(MASK_ALL, val, 16, 32);
  return val;
#endif
}

struct SeedPosition {
  uint32_t seed_index;
  int32_t x;
  int32_t z;
};

constexpr int32_t large_biomes_pos_multiplier = large_biomes ? 4 : 1;

struct alignas(16) MinecraftImprovedNoise {
  uint8_t p[256];
  float xo;
  float yo;
  float zo;
  float pad;
};

struct NoiseOctave {
  MinecraftImprovedNoise noise;
  double freq_mult;
  double amp_mult;
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

__device__ inline XrsrRandomFork seed_fork_xrsr(uint64_t seed) {
  seed ^= XrsrRandom::XRSR_SILVER_RATIO;
  uint64_t low_state = XrsrRandom::mix64(seed);
  uint64_t high_state = XrsrRandom::mix64(seed + XrsrRandom::XRSR_GOLDEN_RATIO);
  
  uint64_t rand_val_1 = XrsrRandom::rol64(low_state + high_state, 17) + low_state;
  
  // update state once
  high_state ^= low_state;
  uint64_t low_state_2 = XrsrRandom::rol64(low_state, 49) ^ high_state ^ (high_state << 21);
  uint64_t high_state_2 = XrsrRandom::rol64(high_state, 28);
  
  // skip state update
  uint64_t rand_val_2 = XrsrRandom::rol64(low_state_2 + high_state_2, 17) + low_state_2;

  return { rand_val_1, rand_val_2 };
}

__device__ inline void double_fork_xrsr(XrsrRandom &rng, XrsrRandomFork &fork_a, XrsrRandomFork &fork_b) {
  uint64_t low_state = rng.lo;
  uint64_t high_state = rng.hi;
  
  // fork A
  uint64_t rand_val_1 = XrsrRandom::rol64(low_state + high_state, 17) + low_state;
  high_state ^= low_state;
  uint64_t low_state_2 = XrsrRandom::rol64(low_state, 49) ^ high_state ^ (high_state << 21);
  uint64_t high_state_2 = XrsrRandom::rol64(high_state, 28);
  
  uint64_t rand_val_2 = XrsrRandom::rol64(low_state_2 + high_state_2, 17) + low_state_2;
  high_state_2 ^= low_state_2;
  uint64_t low_state_3 = XrsrRandom::rol64(low_state_2, 49) ^ high_state_2 ^ (high_state_2 << 21);
  uint64_t high_state_3 = XrsrRandom::rol64(high_state_2, 28);
  
  fork_a = { rand_val_1, rand_val_2 };
  
  // fork B, skip 4th state update
  uint64_t rand_val_3 = XrsrRandom::rol64(low_state_3 + high_state_3, 17) + low_state_3;
  high_state_3 ^= low_state_3;
  uint64_t low_state_4 = XrsrRandom::rol64(low_state_3, 49) ^ high_state_3 ^ (high_state_3 << 21);
  uint64_t high_state_4 = XrsrRandom::rol64(high_state_3, 28);
  
  uint64_t rand_val_4 = XrsrRandom::rol64(low_state_4 + high_state_4, 17) + low_state_4;
  
  fork_b = { rand_val_3, rand_val_4 };
}

// from cubiomes md5 identifiers
constexpr XrsrForkHash HASH_CONTINENTALNESS{
    0x83886c9d0ae3a662, 0xafa638a61b42e8ad}; // md5 "minecraft:continentalness"
constexpr XrsrForkHash HASH_CONTINENTALNESS_LARGE{
    0x9a3f51a113fce8dc,
    0xee2dbd157e5dcdad}; // md5 "minecraft:continentalness_large"
constexpr XrsrForkHash HASH_OCTAVES[]{
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

constexpr auto continentalness_parameters = make_noise_parameters(-9, {1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0});
constexpr auto continentalness_large_parameters = make_noise_parameters(-11, {1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0});

struct OctaveConfig {
  XrsrForkHash fork_hash;
  double freq_mult;
  double amp_mult;
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

  double root_amp_mult = 0.16666666666666666 / (0.1 * (1.0 + 1.0 / amplitudes.size()));

  double freq_mult = 1.0 / (1 << -first_octave);
  double amp_mult = (1 << (amplitudes.size() - 1)) / ((1 << amplitudes.size()) - 1.0) * root_amp_mult;

  for (size_t i = 0; i < amplitudes.size(); i++) {
    res.octaves_a[i] = {HASH_OCTAVES[first_octave + 12 + i], freq_mult, amp_mult * amplitudes[i]};
    res.octaves_b[i] = {HASH_OCTAVES[first_octave + 12 + i], freq_mult * 1.0181268882175227, amp_mult * amplitudes[i]};
    freq_mult *= 2.0;
    amp_mult *= 0.5;
  }

  return res;
}

__device__ constexpr auto continentalness_config = make_normal_noise_config(continentalness_parameters, HASH_CONTINENTALNESS);
__device__ constexpr auto continentalness_large_config = make_normal_noise_config(continentalness_large_parameters, HASH_CONTINENTALNESS_LARGE);
__device__ constexpr auto selected_continentalness_config = large_biomes ? continentalness_large_config : continentalness_config;

// from cubiomes parameters
#include "kernel_0A.h"
#include "kernel_0B.h"

__constant__ float2 c_kernel_0A[6][6][16];
__constant__ float2 c_kernel_0B[6][6][16];

__forceinline__ __device__ float gradient_dot_product(uint8_t hash, float x, float y, float z) {
  uint32_t h = hash & 15;
  float u = h < 8 ? x : y;
  float v = h < 4 ? y : ((h == 12 || h == 14) ? x : z);
  return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

__forceinline__ __device__ float perlin_fade(float value) {
  return value * value * value * fmaf(value, fmaf(value, 6.0f, -15.0f), 10.0f);
}

__forceinline__ __device__ float lerp1d(float progress, float start_val, float end_val) {
  return fmaf(progress, end_val - start_val, start_val);
}

__forceinline__ __device__ float lerp2d(float progress_x, float progress_y, float val_00, float val_10, float val_01, float val_11) {
  return lerp1d(progress_y, lerp1d(progress_x, val_00, val_10), lerp1d(progress_x, val_01, val_11));
}

__forceinline__ __device__ float lerp3d(float progress_x, float progress_y, float progress_z, float val_000, float val_100, float val_010, float val_110, float val_001, float val_101, float val_011, float val_111) {
  return lerp1d(progress_z, lerp2d(progress_x, progress_y, val_000, val_100, val_010, val_110), lerp2d(progress_x, progress_y, val_001, val_101, val_011, val_111));
}

__forceinline__ __device__ void compute_perlin_coords(float val, int32_t &grid, float &frac, float &fade) {
  grid = __float2int_rd(val);
  frac = val - (float)grid;
  fade = perlin_fade(frac);
}

__device__ __forceinline__ void compute_cell(
    const MinecraftImprovedNoise &noise, int32_t int_x, int32_t int_y, int32_t int_z,
    uint8_t &corner_000, uint8_t &corner_100, uint8_t &corner_010, uint8_t &corner_110,
    uint8_t &corner_001, uint8_t &corner_101, uint8_t &corner_011, uint8_t &corner_111)
{
  const int32_t y_plus_1 = int_y + 1;
  const int32_t z_plus_1 = int_z + 1;
  uint8_t p0  = noise.p[int_x & 0xFF];
  uint8_t p1  = noise.p[(int_x + 1) & 0xFF];
  uint8_t p00 = noise.p[(p0 + int_y) & 0xFF];
  uint8_t p01 = noise.p[(p0 + y_plus_1) & 0xFF];
  uint8_t p10 = noise.p[(p1 + int_y) & 0xFF];
  uint8_t p11 = noise.p[(p1 + y_plus_1) & 0xFF];
  corner_000 = noise.p[(p00 + int_z) & 0xFF];
  corner_100 = noise.p[(p10 + int_z) & 0xFF];
  corner_010 = noise.p[(p01 + int_z) & 0xFF];
  corner_110 = noise.p[(p11 + int_z) & 0xFF];
  corner_001 = noise.p[(p00 + z_plus_1) & 0xFF];
  corner_101 = noise.p[(p10 + z_plus_1) & 0xFF];
  corner_011 = noise.p[(p01 + z_plus_1) & 0xFF];
  corner_111 = noise.p[(p11 + z_plus_1) & 0xFF];
}

__device__ __forceinline__ float trilinear_interp(
    float frac_x, float frac_y, float frac_z,
    float fade_x, float fade_y, float fade_z,
    uint8_t corner_000, uint8_t corner_100, uint8_t corner_010, uint8_t corner_110,
    uint8_t corner_001, uint8_t corner_101, uint8_t corner_011, uint8_t corner_111)
{
  float noise_000 = gradient_dot_product(corner_000, frac_x, frac_y, frac_z);
  float noise_100 = gradient_dot_product(corner_100, frac_x - 1.0f, frac_y, frac_z);
  float noise_010 = gradient_dot_product(corner_010, frac_x, frac_y - 1.0f, frac_z);
  float noise_110 = gradient_dot_product(corner_110, frac_x - 1.0f, frac_y - 1.0f, frac_z);
  float noise_001 = gradient_dot_product(corner_001, frac_x, frac_y, frac_z - 1.0f);
  float noise_101 = gradient_dot_product(corner_101, frac_x - 1.0f, frac_y, frac_z - 1.0f);
  float noise_011 = gradient_dot_product(corner_011, frac_x, frac_y - 1.0f, frac_z - 1.0f);
  float noise_111 = gradient_dot_product(corner_111, frac_x - 1.0f, frac_y - 1.0f, frac_z - 1.0f);
  return lerp3d(fade_x, fade_y, fade_z, noise_000, noise_100, noise_010, noise_110, noise_001, noise_101, noise_011, noise_111);
}

__device__ float sample_minecraft_noise(const MinecraftImprovedNoise &noise, float x, float y, float z) {
  x += noise.xo;
  y += noise.yo;
  z += noise.zo;

  int32_t int_x, int_y, int_z;
  float frac_x, frac_y, frac_z, fade_x, fade_y, fade_z;

  compute_perlin_coords(x, int_x, frac_x, fade_x);
  compute_perlin_coords(y, int_y, frac_y, fade_y);
  compute_perlin_coords(z, int_z, frac_z, fade_z);

  uint8_t c000, c100, c010, c110, c001, c101, c011, c111;
  compute_cell(noise, int_x, int_y, int_z, c000, c100, c010, c110, c001, c101, c011, c111);

  return trilinear_interp(frac_x, frac_y, frac_z, fade_x, fade_y, fade_z, c000, c100, c010, c110, c001, c101, c011, c111);
}

template <OctaveConfig config>
__forceinline__ __device__ float sample_noise_octave(const MinecraftImprovedNoise &noise, int32_t x, int32_t y, int32_t z) {
  return sample_minecraft_noise(noise, x * (float)config.freq_mult, y * (float)config.freq_mult, z * (float)config.freq_mult) * (float)config.amp_mult;
}

__device__ void initialize_minecraft_noise(MinecraftImprovedNoise &noise, XrsrRandom &&random) {
  noise.xo = random.nextFloat() * 256.0f;
  noise.yo = random.nextFloat() * 256.0f;
  noise.zo = random.nextFloat() * 256.0f;
  noise.pad = 0.0f;

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

constexpr float GRADIENT_VECTORS_1_PREFILTER_THRESHOLD = -12.0f;
constexpr float GRADIENT_VECTORS_1_FINAL_THRESHOLD      = -18.5f;

constexpr float GRADIENT_VECTORS_2_PREFILTER_THRESHOLD = -13.5f;
constexpr float GRADIENT_VECTORS_2_FINAL_THRESHOLD      = -20.0f;

template <typename IndexT>
__device__ __forceinline__ float eval_conv_center_2x2(
    const float conv_z0[6][256],
    const float conv_z1[6][256],
    const IndexT* idx0,
    const IndexT* idx1)
{
  return
      conv_z0[2][(idx0[2]) & 0xFF] +
      conv_z0[3][(idx0[3]) & 0xFF] +
      conv_z1[2][(idx1[2]) & 0xFF] +
      conv_z1[3][(idx1[3]) & 0xFF];
}

template <typename IndexT>
__device__ __forceinline__ float eval_conv_edges_12(
    const float conv_z0[6][256],
    const float conv_z1[6][256],
    const IndexT* idx0,
    const IndexT* idx1)
{
  return
      conv_z0[0][(idx0[0]) & 0xFF] +
      conv_z1[0][(idx1[0]) & 0xFF] +
      conv_z0[1][(idx0[1]) & 0xFF] +
      conv_z1[1][(idx1[1]) & 0xFF] +
      conv_z0[4][(idx0[4]) & 0xFF] +
      conv_z1[4][(idx1[4]) & 0xFF] +
      conv_z0[5][(idx0[5]) & 0xFF] +
      conv_z1[5][(idx1[5]) & 0xFF];
}

template <typename TableT>
__device__ __forceinline__ void compute_z_conv(
    const TableT& c_kernel_table,
    const MinecraftImprovedNoise &noise,
    int32_t base_z_index,
    float conv_z0[6][256],
    float conv_z1[6][256]) 
{
  uint8_t p_z[6];
#pragma unroll
  for (int32_t offset_z = 0; offset_z < 6; ++offset_z) {
    p_z[offset_z] = noise.p[(base_z_index + offset_z) & 0xFF] & 0xF;
  }
#pragma unroll
  for (int32_t offset_x = 0; offset_x < 6; ++offset_x) {
    float conv0 = 0.0f;
    float conv1 = 0.0f;
#pragma unroll
    for (int32_t offset_z = 0; offset_z < 6; ++offset_z) {
      const uint32_t p = p_z[offset_z];
      const float2 val = c_kernel_table[offset_x][offset_z][p];
      conv0 += val.x;
      conv1 += val.y;
    }
    conv_z0[offset_x][base_z_index] = conv0;
    conv_z1[offset_x][base_z_index] = conv1;
  }
}

void initialize_convolution_kernels() {
  auto transpose_kernel = [](float2 dst[6][6][16], const float src[2][6][6][16]) {
    for (int offset_y = 0; offset_y < 2; ++offset_y)
      for (int offset_x = 0; offset_x < 6; ++offset_x)
        for (int offset_z = 0; offset_z < 6; ++offset_z)
          for (int p = 0; p < 16; ++p) {
            if (offset_y == 0) dst[offset_x][offset_z][p].x = src[offset_y][offset_x][offset_z][p];
            else          dst[offset_x][offset_z][p].y = src[offset_y][offset_x][offset_z][p];
          }
  };

  float2 transposed_kernels[6][6][16];

  transpose_kernel(transposed_kernels, host_kernel_0A);
  TRY_HIP(hipMemcpyToSymbol(c_kernel_0A, transposed_kernels, sizeof(transposed_kernels)));

  transpose_kernel(transposed_kernels, host_kernel_0B);
  TRY_HIP(hipMemcpyToSymbol(c_kernel_0B, transposed_kernels, sizeof(transposed_kernels)));
}

struct DeviceBuffer {
  void *data;
  size_t size;

  DeviceBuffer(size_t size) : size(size) { TRY_HIP(hipMalloc(&data, size)); }

  ~DeviceBuffer() { TRY_HIP(hipFree(data)); }
};

template <typename T> struct OutputBuffer {
  T *data;
  uint32_t *len;
  uint32_t max_len;

  __host__ __device__ OutputBuffer(T *data, uint32_t *len, uint32_t max_len)
      : data(data), len(len), max_len(max_len) {}

  __host__ OutputBuffer(const DeviceBuffer &buffer, uint32_t *len)
      : data((T *)buffer.data), len(len), max_len(buffer.size / sizeof(T)) {}

  __host__ __device__ OutputBuffer(const OutputBuffer<T> &other)
      : data(other.data), len(other.len), max_len(other.max_len) {}
};

template <typename T> struct InputBuffer {
  const T *data;
  const uint32_t *len;

  __host__ __device__ InputBuffer(const T *data, const uint32_t *len) : data(data), len(len) {}

  __host__ __device__ InputBuffer(const OutputBuffer<T> &buffer)
      : data(buffer.data), len(buffer.len) {}

  __host__ __device__ InputBuffer(const InputBuffer<T> &other) : data(other.data), len(other.len) {}
};

__global__ void cap_length_gpu_kernel(uint32_t* length, uint32_t max_length) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (*length > max_length) {
      *length = max_length;
    }
  }
}

void cap_gpu_buffer_length(uint32_t* dev_length, uint32_t max_length, hipStream_t stream) {
  cap_length_gpu_kernel<<<1, 1, 0, stream>>>(dev_length, max_length);
  TRY_HIP(hipGetLastError());
}

struct BaseStage {
  std::string name;
  uint32_t *host_input_count;
  uint32_t *host_output_count;
  uint64_t input_multiplier;
  uint32_t maximum_buffer_capacity;
  HipEventWrapper event;
  double total_time = 0;
  uint64_t total_inputs = 0;
  uint64_t total_outputs = 0;

  BaseStage(std::string name, uint32_t *inputs_len, uint32_t *outputs_len, uint64_t multiplier, uint32_t max_outputs_len)
      : name(std::move(name)), host_input_count(inputs_len), host_output_count(outputs_len), input_multiplier(multiplier), maximum_buffer_capacity(max_outputs_len) {}

  virtual ~BaseStage() = default;
  virtual void cap_len() = 0;

  void record(hipStream_t stream = 0) { event.record(stream); }

  void update(HipEventWrapper &prev_event) {
    total_time += prev_event.elapsed(event) * 1e-3;
    total_inputs += host_input_count ? *host_input_count : 1;
    total_outputs += *host_output_count;
    if (*host_output_count > maximum_buffer_capacity) {
      std::printf("%s outputs overflow: len = %" PRIu32 " max_len = %" PRIu32 "\n", name.c_str(), *host_output_count, maximum_buffer_capacity);
    }
  }

  void reset() {
    total_time = 0;
    total_inputs = 0;
    total_outputs = 0;
  }
};

template <typename T>
struct Stage : public BaseStage {
  OutputBuffer<T> output_buffer;

  Stage(std::vector<BaseStage*> &pipeline, std::string name, DeviceBuffer &db, uint32_t *dev_len, uint32_t *host_len, uint32_t *inputs_len, uint64_t multiplier)
      : BaseStage(name, inputs_len, host_len, multiplier, db.size / sizeof(T)), output_buffer(db, dev_len) {
    pipeline.push_back(this);
  }

  void cap_len() override {
    *host_output_count = std::min(*host_output_count, maximum_buffer_capacity);
  }
};

struct CustomStage : public BaseStage {
  CustomStage(std::vector<BaseStage*> &pipeline, std::string name, uint32_t *inputs_len, uint32_t *outputs_len, uint64_t multiplier, uint32_t max_outputs_len)
      : BaseStage(name, inputs_len, outputs_len, multiplier, max_outputs_len) {
    pipeline.push_back(this);
  }

  void cap_len() override {}
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

struct BufferLengths {
  uint32_t prefilter_seeds_count;
  uint32_t gradient_vectors_1_count;
  uint32_t gradient_vectors_2_count;
  uint32_t coarse_octave_b_count;
  uint32_t fine_seeds_count;
  uint32_t refinement_counts[7];
};

__global__ void gather_winning_seeds_kernel(
    const uint64_t* __restrict__ seeds,
    const SeedPosition* __restrict__ positions,
    uint64_t* __restrict__ out_seeds,
    uint32_t count)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out_seeds[i] = seeds[positions[i].seed_index];
  }
}

constexpr uint32_t GATHER_THREADS_PER_BLOCK = 256;

namespace NoiseCacheInitializer {
constexpr uint32_t MAX_CACHE_SEEDS = UINT64_C(1) << 16;
constexpr uint32_t THREADS_PER_BLOCK = 32;

struct NoiseCache {
  MinecraftImprovedNoise octaves[18];

  template <size_t Octaves, size_t Idx = 0>
  __device__ __forceinline__ float sample(int32_t x, int32_t y, int32_t z) const {
    if constexpr (Idx >= Octaves) {
      return 0.0f;
    } else {
      constexpr size_t I = Idx / 2;
      if constexpr (I >= 9) {
        return 0.0f;
      } else {
        if constexpr (Idx % 2 == 0) {
          return sample_noise_octave<selected_continentalness_config.octaves_a[I]>(octaves[Idx], x, y, z)
                 + sample<Octaves, Idx + 1>(x, y, z);
        } else {
          return sample_noise_octave<selected_continentalness_config.octaves_b[I]>(octaves[Idx], x, y, z)
                 + sample<Octaves, Idx + 1>(x, y, z);
        }
      }
    }
  }

  template <size_t Octaves, size_t I = 0>
  __device__ __forceinline__ float sample_only_a(int32_t x, int32_t y, int32_t z) const {
    if constexpr (2 * I >= Octaves || I >= 9) {
      return 0.0f;
    } else {
      return sample_noise_octave<selected_continentalness_config.octaves_a[I]>(octaves[2 * I], x, y, z) 
             + sample_only_a<Octaves, I + 1>(x, y, z);
    }
  }
};

template <uint32_t TPB>
__device__ __forceinline__ void copy_noise_to_global_memory(MinecraftImprovedNoise (&shared_noise)[TPB], NoiseCache *cache_buffers, uint32_t octave_index, uint32_t block_base, uint32_t input_len, bool active) {
  if (!active) {
    return;
  }

  constexpr uint32_t uint4_words_per_struct = sizeof(MinecraftImprovedNoise) / sizeof(uint4);
  const uint4 *src_flat = reinterpret_cast<const uint4 *>(&shared_noise[threadIdx.x]);
  MinecraftImprovedNoise &dst = cache_buffers[block_base + threadIdx.x].octaves[octave_index];
  uint4 *dst_flat = reinterpret_cast<uint4 *>(&dst);

#pragma unroll
  for (uint32_t word_idx = 0; word_idx < uint4_words_per_struct; ++word_idx) {
    dst_flat[word_idx] = src_flat[word_idx];
  }
}

template <uint32_t TPB>
__device__ __forceinline__ void copy_noise_to_global_memory_fine(MinecraftImprovedNoise (&shared_noise)[TPB], NoiseCache *cache_buffers, uint32_t octave_index, uint32_t block_base, uint32_t input_len, const uint32_t *compact_seeds, bool active) {
  if (!active) {
    return;
  }

  constexpr uint32_t uint4_words_per_struct = sizeof(MinecraftImprovedNoise) / sizeof(uint4);
  const uint32_t seed_idx = compact_seeds[block_base + threadIdx.x];
  const uint4 *src_flat = reinterpret_cast<const uint4 *>(&shared_noise[threadIdx.x]);
  MinecraftImprovedNoise &dst = cache_buffers[seed_idx].octaves[octave_index];
  uint4 *dst_flat = reinterpret_cast<uint4 *>(&dst);

#pragma unroll
  for (uint32_t word_idx = 0; word_idx < uint4_words_per_struct; ++word_idx) {
    dst_flat[word_idx] = src_flat[word_idx];
  }
}

__device__ void init_octave(const XrsrRandomFork &noise_fork, const XrsrForkHash &fork_hash, NoiseCache *results, uint32_t octave_index, uint32_t block_base, uint32_t input_len, bool active) {
  __shared__ alignas(16) MinecraftImprovedNoise shared_noise[THREADS_PER_BLOCK];

  if (active) {
    initialize_minecraft_noise(shared_noise[threadIdx.x], noise_fork.from(fork_hash));
  }

  copy_noise_to_global_memory<THREADS_PER_BLOCK>(shared_noise, results, octave_index, block_base, input_len, active);
  __syncthreads();
}

__device__ void init_octave_fine(const XrsrRandomFork &noise_fork, const XrsrForkHash &fork_hash, NoiseCache *results, uint32_t octave_index, uint32_t block_base, uint32_t input_len, bool active, const uint32_t *compact_seeds) {
  __shared__ alignas(16) MinecraftImprovedNoise shared_noise[THREADS_PER_BLOCK];

  if (active) {
    initialize_minecraft_noise(shared_noise[threadIdx.x], noise_fork.from(fork_hash));
  }

  copy_noise_to_global_memory_fine<THREADS_PER_BLOCK>(shared_noise, results, octave_index, block_base, input_len, compact_seeds, active);
  __syncthreads();
}

template <size_t... Is>
__device__ __forceinline__ void init_octaves_a_helper(const XrsrRandomFork &noise_a_fork, NoiseCache *results, uint32_t block_base, uint32_t input_len, bool active, std::index_sequence<Is...>) {
  (init_octave(noise_a_fork, selected_continentalness_config.octaves_a[Is].fork_hash, results, 2 * Is, block_base, input_len, active), ...);
}

template <size_t... Is>
__device__ __forceinline__ void init_octaves_b_helper(const XrsrRandomFork &noise_b_fork, NoiseCache *results, uint32_t block_base, uint32_t input_len, bool active, std::index_sequence<Is...>) {
  (init_octave(noise_b_fork, selected_continentalness_config.octaves_b[Is].fork_hash, results, 2 * Is + 1, block_base, input_len, active), ...);
}

template <size_t... Is>
__device__ __forceinline__ void init_octaves_a_helper_fine(const XrsrRandomFork &noise_a_fork, NoiseCache *results, uint32_t block_base, uint32_t input_len, bool active, const uint32_t *compact_seeds, std::index_sequence<Is...>) {
  (init_octave_fine(noise_a_fork, selected_continentalness_config.octaves_a[Is + 2].fork_hash, results, 2 * (Is + 2), block_base, input_len, active, compact_seeds), ...);
}

template <size_t... Is>
__device__ __forceinline__ void init_octaves_b_helper_fine(const XrsrRandomFork &noise_b_fork, NoiseCache *results, uint32_t block_base, uint32_t input_len, bool active, const uint32_t *compact_seeds, std::index_sequence<Is...>) {
  (init_octave_fine(noise_b_fork, selected_continentalness_config.octaves_b[Is + 2].fork_hash, results, 2 * (Is + 2) + 1, block_base, input_len, active, compact_seeds), ...);
}

__global__ void deduplicate_seeds_kernel(InputBuffer<SeedPosition> coarse_outputs, uint32_t *flags, uint32_t *unique_seeds, uint32_t *unique_count) {
  uint32_t len = *coarse_outputs.len;
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len; idx += gridDim.x * blockDim.x) {
    uint32_t seed_idx = coarse_outputs.data[idx].seed_index;
    if (atomicExch(&flags[seed_idx], 1) == 0) {
      uint32_t pos = atomicAdd(unique_count, 1);
      unique_seeds[pos] = seed_idx;
    }
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK) void kernel_coarse(InputBuffer<uint64_t> input, NoiseCache *results) {
  uint32_t block_base = blockIdx.x * blockDim.x;
  uint32_t input_len = *input.len;
  if (block_base >= input_len) {
    return;
  }

  uint32_t index = block_base + threadIdx.x;
  bool active = (index < input_len);

  XrsrRandomFork noise_a_fork{};
  XrsrRandomFork noise_b_fork{};

  if (active) {
    uint64_t seed = input.data[index];
    const auto seed_fork = seed_fork_xrsr(seed);
    auto noise_random = seed_fork.from(selected_continentalness_config.fork_hash);
    double_fork_xrsr(noise_random, noise_a_fork, noise_b_fork);
  }

  init_octaves_a_helper(noise_a_fork, results, block_base, input_len, active, std::make_index_sequence<2>{});
  init_octaves_b_helper(noise_b_fork, results, block_base, input_len, active, std::make_index_sequence<2>{});
}

__global__ __launch_bounds__(THREADS_PER_BLOCK) void kernel_fine(InputBuffer<uint64_t> input, NoiseCache *results, const uint32_t *compact_seeds, const uint32_t *compact_count) {
  uint32_t block_base = blockIdx.x * blockDim.x;
  uint32_t input_len = *compact_count;
  if (block_base >= input_len) {
    return;
  }

  uint32_t index = block_base + threadIdx.x;
  bool active = (index < input_len);

  XrsrRandomFork noise_a_fork{};
  XrsrRandomFork noise_b_fork{};

  if (active) {
    uint32_t seed_idx = compact_seeds[index];
    uint64_t seed = input.data[seed_idx];
    const auto seed_fork = seed_fork_xrsr(seed);
    auto noise_random = seed_fork.from(selected_continentalness_config.fork_hash);
    double_fork_xrsr(noise_random, noise_a_fork, noise_b_fork);
  }

  init_octaves_a_helper_fine(noise_a_fork, results, block_base, input_len, active, compact_seeds, std::make_index_sequence<7>{});
  init_octaves_b_helper_fine(noise_b_fork, results, block_base, input_len, active, compact_seeds, std::make_index_sequence<7>{});
}

void run_coarse(InputBuffer<uint64_t> input, NoiseCache *noise_caches, hipStream_t stream) {
  uint32_t blocks = MAX_CACHE_SEEDS / THREADS_PER_BLOCK;
  kernel_coarse<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(input, noise_caches);
  TRY_HIP(hipGetLastError());
}

void run_fine(InputBuffer<uint64_t> input, NoiseCache *noise_caches, InputBuffer<SeedPosition> coarse_outputs, uint32_t *flags, uint32_t *compact_seeds, uint32_t *compact_count, hipStream_t stream) {
  TRY_HIP(hipMemsetAsync(flags, 0, MAX_CACHE_SEEDS * sizeof(uint32_t), stream));
  
  deduplicate_seeds_kernel<<<256, 256, 0, stream>>>(coarse_outputs, flags, compact_seeds, compact_count);
  TRY_HIP(hipGetLastError());

  uint32_t blocks = MAX_CACHE_SEEDS / THREADS_PER_BLOCK;
  kernel_fine<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(input, noise_caches, compact_seeds, compact_count);
  TRY_HIP(hipGetLastError());
}

} // namespace NoiseCacheInitializer

namespace SeedPreFilter {
constexpr uint32_t THREADS_PER_BLOCK = 256;
constexpr uint32_t SEED_BATCH_SIZE = UINT64_C(1) << 28;

constexpr XrsrForkHash hash_for_y_offset_fork(XrsrForkHash hash) {
  XrsrRandom rng{hash.lo, hash.hi};
  rng.nextInternal();
  return {rng.lo, rng.hi};
}

__device__ constexpr XrsrForkHash y_hash_a0 = hash_for_y_offset_fork(selected_continentalness_config.octaves_a[0].fork_hash);
__device__ constexpr XrsrForkHash y_hash_a1 = hash_for_y_offset_fork(selected_continentalness_config.octaves_a[1].fork_hash);
__device__ constexpr XrsrForkHash y_hash_a2 = hash_for_y_offset_fork(selected_continentalness_config.octaves_a[2].fork_hash);
__device__ constexpr XrsrForkHash y_hash_b0 = hash_for_y_offset_fork(selected_continentalness_config.octaves_b[0].fork_hash);
__device__ constexpr XrsrForkHash y_hash_b1 = hash_for_y_offset_fork(selected_continentalness_config.octaves_b[1].fork_hash);
__device__ constexpr XrsrForkHash y_hash_b2 = hash_for_y_offset_fork(selected_continentalness_config.octaves_b[2].fork_hash);

__device__ __forceinline__ uint32_t compute_octave_y_offset_score_inline(uint64_t lo, uint64_t hi, const XrsrForkHash &hash) {
  uint64_t l = lo ^ hash.lo;
  uint64_t h = hi ^ hash.hi;
  uint64_t r = XrsrRandom::rol64(l + h, 17) + l;
  int32_t r_24 = (r >> 32) & 0xFFFFFF;
  return abs(r_24 - 0x800000);
}

__device__ __forceinline__ void advance_xrsr(uint64_t &l, uint64_t &h) {
  h ^= l;
  uint64_t temp_l = XrsrRandom::rol64(l, 49) ^ h ^ (h << 21);
  h = XrsrRandom::rol64(h, 28);
  l = temp_l;
}

__global__ __launch_bounds__(THREADS_PER_BLOCK) void kernel(uint64_t start_seed, uint32_t count, OutputBuffer<uint64_t> outputs) {
  constexpr uint32_t MAX_SCORE = 63753419;

  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  uint64_t seed = start_seed + index;

  uint64_t seed_mix = seed ^ XrsrRandom::XRSR_SILVER_RATIO;
  uint64_t low_state = XrsrRandom::mix64(seed_mix);
  uint64_t high_state = XrsrRandom::mix64(seed_mix + XrsrRandom::XRSR_GOLDEN_RATIO);
  
  uint64_t seed_fork_lo = XrsrRandom::rol64(low_state + high_state, 17) + low_state;
  high_state ^= low_state;
  uint64_t l2 = XrsrRandom::rol64(low_state, 49) ^ high_state ^ (high_state << 21);
  uint64_t h2 = XrsrRandom::rol64(high_state, 28);
  uint64_t seed_fork_hi = XrsrRandom::rol64(l2 + h2, 17) + l2;

  uint64_t noise_l = seed_fork_lo ^ selected_continentalness_config.fork_hash.lo;
  uint64_t noise_h = seed_fork_hi ^ selected_continentalness_config.fork_hash.hi;

  uint64_t fork_a_l = XrsrRandom::rol64(noise_l + noise_h, 17) + noise_l;
  advance_xrsr(noise_l, noise_h);
  uint64_t fork_a_h = XrsrRandom::rol64(noise_l + noise_h, 17) + noise_l;
  
  uint64_t a_y_h = fork_a_h ^ fork_a_l;
  uint64_t a_y_lo = XrsrRandom::rol64(fork_a_l, 49) ^ a_y_h ^ (a_y_h << 21);
  uint64_t a_y_hi = XrsrRandom::rol64(a_y_h, 28);

  uint32_t score = 0;
  score += 35 * compute_octave_y_offset_score_inline(a_y_lo, a_y_hi, y_hash_a0);
  if (score >= MAX_SCORE) return;
  score += 11 * compute_octave_y_offset_score_inline(a_y_lo, a_y_hi, y_hash_a1);
  if (score >= MAX_SCORE) return;
  score += 4 * compute_octave_y_offset_score_inline(a_y_lo, a_y_hi, y_hash_a2);
  if (score >= MAX_SCORE) return;

  advance_xrsr(noise_l, noise_h);
  uint64_t fork_b_l = XrsrRandom::rol64(noise_l + noise_h, 17) + noise_l;
  advance_xrsr(noise_l, noise_h);
  uint64_t fork_b_h = XrsrRandom::rol64(noise_l + noise_h, 17) + noise_l;

  uint64_t b_y_h = fork_b_h ^ fork_b_l;
  uint64_t b_y_lo = XrsrRandom::rol64(fork_b_l, 49) ^ b_y_h ^ (b_y_h << 21);
  uint64_t b_y_hi = XrsrRandom::rol64(b_y_h, 28);

  score += 35 * compute_octave_y_offset_score_inline(b_y_lo, b_y_hi, y_hash_b0);
  if (score >= MAX_SCORE) return;
  score += 11 * compute_octave_y_offset_score_inline(b_y_lo, b_y_hi, y_hash_b1);
  if (score >= MAX_SCORE) return;
  score += 4 * compute_octave_y_offset_score_inline(b_y_lo, b_y_hi, y_hash_b2);
  if (score >= MAX_SCORE) return;

  uint32_t result_index = atomicAdd(outputs.len, 1);
  if (result_index < outputs.max_len){
    outputs.data[result_index] = seed;
  }
}

void run(uint64_t start_seed, uint32_t count, OutputBuffer<uint64_t> outputs, hipStream_t stream) {
  uint32_t blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  if (blocks > 0) {
    kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(start_seed, count, outputs);
    TRY_HIP(hipGetLastError());
    cap_gpu_buffer_length(outputs.len, outputs.max_len, stream);
  }
}
} // namespace SeedPreFilter

namespace GridGradientFilter1 {
constexpr uint32_t THREADS_PER_BLOCK = 256;

__global__
__launch_bounds__(THREADS_PER_BLOCK) void kernel(
    const InputBuffer<uint64_t> seeds,
    OutputBuffer<SeedPosition> outputs,
    const NoiseCacheInitializer::NoiseCache* __restrict__ noise_caches)
{
  __shared__ float2 s_kernel_0A[6][6][16];
  for (uint32_t i = threadIdx.x; i < 576; i += blockDim.x) {
    ((float2*)s_kernel_0A)[i] = ((const float2*)c_kernel_0A)[i];
  }
  __syncthreads();

  __shared__ alignas(16) MinecraftImprovedNoise octave_0A;
  __shared__ float conv_z0[6][256];
  __shared__ float conv_z1[6][256];
  __shared__ alignas(16) uint8_t xy_permutation_cache[2][272];

  const int32_t local_z_coord = threadIdx.x;

  const uint32_t seeds_len = *seeds.len;
  for (uint32_t seed_index = blockIdx.x; seed_index < seeds_len; seed_index += gridDim.x) {
    __syncthreads();

    if (local_z_coord < 17) {
      reinterpret_cast<uint4*>(&octave_0A)[local_z_coord] =
          reinterpret_cast<const uint4*>(&noise_caches[seed_index].octaves[0])[local_z_coord];
    }

    __syncthreads();

    compute_z_conv(s_kernel_0A, octave_0A, local_z_coord, conv_z0, conv_z1);

    const int32_t cell_size = 512 * large_biomes_pos_multiplier;
    const int32_t x_center = (2.5f - octave_0A.xo) * cell_size;
    const int32_t ny = octave_0A.yo;
    const int32_t z_center = (2.5f - octave_0A.zo) * cell_size;

    const uint8_t idx_x = octave_0A.p[local_z_coord];
    const uint8_t v0 = octave_0A.p[(idx_x + ny) & 0xFF];
    const uint8_t v1 = octave_0A.p[(idx_x + ny + 1) & 0xFF];
    xy_permutation_cache[0][local_z_coord] = v0;
    xy_permutation_cache[1][local_z_coord] = v1;

    if (local_z_coord < 6) {
      xy_permutation_cache[0][256 + local_z_coord] = v0;
      xy_permutation_cache[1][256 + local_z_coord] = v1;
    }

    __syncthreads();

    int32_t x = x_center;
    const int32_t z = z_center + local_z_coord * cell_size;

    uchar4 xy_perm_word0_A = *reinterpret_cast<const uchar4*>(&xy_permutation_cache[0][0]);
    uchar4 xy_perm_word1_A = *reinterpret_cast<const uchar4*>(&xy_permutation_cache[0][4]);
    uchar4 xy_perm_word0_B = *reinterpret_cast<const uchar4*>(&xy_permutation_cache[1][0]);
    uchar4 xy_perm_word1_B = *reinterpret_cast<const uchar4*>(&xy_permutation_cache[1][4]);

    #pragma unroll 4
    for (int32_t nx = 0; nx < 256; nx += 8) {
      uchar4 xy_perm_word2_A = *reinterpret_cast<const uchar4*>(&xy_permutation_cache[0][nx + 8]);
      uchar4 xy_perm_word3_A = *reinterpret_cast<const uchar4*>(&xy_permutation_cache[0][nx + 12]);
      uchar4 xy_perm_word2_B = *reinterpret_cast<const uchar4*>(&xy_permutation_cache[1][nx + 8]);
      uchar4 xy_perm_word3_B = *reinterpret_cast<const uchar4*>(&xy_permutation_cache[1][nx + 12]);

      uint32_t noise_grid_coords_A[13];
      noise_grid_coords_A[0]  = xy_perm_word0_A.x + local_z_coord;
      noise_grid_coords_A[1]  = xy_perm_word0_A.y + local_z_coord;
      noise_grid_coords_A[2]  = xy_perm_word0_A.z + local_z_coord;
      noise_grid_coords_A[3]  = xy_perm_word0_A.w + local_z_coord;
      noise_grid_coords_A[4]  = xy_perm_word1_A.x + local_z_coord;
      noise_grid_coords_A[5]  = xy_perm_word1_A.y + local_z_coord;
      noise_grid_coords_A[6]  = xy_perm_word1_A.z + local_z_coord;
      noise_grid_coords_A[7]  = xy_perm_word1_A.w + local_z_coord;
      noise_grid_coords_A[8]  = xy_perm_word2_A.x + local_z_coord;
      noise_grid_coords_A[9]  = xy_perm_word2_A.y + local_z_coord;
      noise_grid_coords_A[10] = xy_perm_word2_A.z + local_z_coord;
      noise_grid_coords_A[11] = xy_perm_word2_A.w + local_z_coord;
      noise_grid_coords_A[12] = xy_perm_word3_A.x + local_z_coord;

      uint32_t noise_grid_coords_B[13];
      noise_grid_coords_B[0]  = xy_perm_word0_B.x + local_z_coord;
      noise_grid_coords_B[1]  = xy_perm_word0_B.y + local_z_coord;
      noise_grid_coords_B[2]  = xy_perm_word0_B.z + local_z_coord;
      noise_grid_coords_B[3]  = xy_perm_word0_B.w + local_z_coord;
      noise_grid_coords_B[4]  = xy_perm_word1_B.x + local_z_coord;
      noise_grid_coords_B[5]  = xy_perm_word1_B.y + local_z_coord;
      noise_grid_coords_B[6]  = xy_perm_word1_B.z + local_z_coord;
      noise_grid_coords_B[7]  = xy_perm_word1_B.w + local_z_coord;
      noise_grid_coords_B[8]  = xy_perm_word2_B.x + local_z_coord;
      noise_grid_coords_B[9]  = xy_perm_word2_B.y + local_z_coord;
      noise_grid_coords_B[10] = xy_perm_word2_B.z + local_z_coord;
      noise_grid_coords_B[11] = xy_perm_word2_B.w + local_z_coord;
      noise_grid_coords_B[12] = xy_perm_word3_B.x + local_z_coord;

#pragma unroll
      for (int candidate = 0; candidate < 8; ++candidate) {
        const uint32_t* cw0 = &noise_grid_coords_A[candidate];
        const uint32_t* cw1 = &noise_grid_coords_B[candidate];

        const float gate = eval_conv_center_2x2(conv_z0, conv_z1, cw0, cw1);
        if (gate >= GRADIENT_VECTORS_1_PREFILTER_THRESHOLD) {
          const float score = gate + eval_conv_edges_12(conv_z0, conv_z1, cw0, cw1);
          if (score > GRADIENT_VECTORS_1_FINAL_THRESHOLD) {
            uint32_t res_idx = atomicAdd(outputs.len, 1);
            if (res_idx < outputs.max_len) {
              outputs.data[res_idx] = {seed_index, x + candidate * cell_size, z};
            }
          }
        }
      }

      x += 8 * cell_size;

      xy_perm_word0_A = xy_perm_word2_A;
      xy_perm_word1_A = xy_perm_word3_A;
      xy_perm_word0_B = xy_perm_word2_B;
      xy_perm_word1_B = xy_perm_word3_B;
    }
  }
}

void run(
    const InputBuffer<uint64_t> seeds,
    OutputBuffer<SeedPosition> outputs,
    const NoiseCacheInitializer::NoiseCache* __restrict__ noise_caches,
    hipStream_t stream)
{
  kernel<<<2048, THREADS_PER_BLOCK, 0, stream>>>(seeds, outputs, noise_caches);
  TRY_HIP(hipGetLastError());
  cap_gpu_buffer_length(outputs.len, outputs.max_len, stream);
}
} // namespace GridGradientFilter1

namespace MultiOctaveRefinementFilter {
template <int32_t NOISE_THRESHOLD, size_t OCTAVES, uint32_t POSITION_RANGE, uint32_t SAMPLES, uint32_t MIN_COUNT, bool FLIPPED_SPARSE_SAMPLES, bool MOVE_CENTER, bool ONLY_A_OCTAVES>
struct RefinementFilterTemplate {
  static constexpr float noise_threshold = NOISE_THRESHOLD / 10000.0f;
  static constexpr size_t octaves = OCTAVES;
  static constexpr uint32_t position_range = POSITION_RANGE;
  static constexpr uint32_t samples = SAMPLES;
  static constexpr uint32_t min_count = MIN_COUNT;
  static constexpr bool flipped_sparse_samples = FLIPPED_SPARSE_SAMPLES;
  static constexpr bool move_center = MOVE_CENTER;
  static constexpr bool only_a_octaves = ONLY_A_OCTAVES;

  static constexpr uint32_t THREADS_PER_BLOCK = 256;
  static_assert(samples >= 32 && samples <= THREADS_PER_BLOCK * THREADS_PER_BLOCK && is_power_of_two(samples));
  static constexpr uint32_t samples_square_size = UINT32_C(1) << (compile_time_log2(samples) + 1) / 2;
  static constexpr bool samples_square_sparse = compile_time_log2(samples) % 2 == 1;
  static_assert(!flipped_sparse_samples || samples_square_sparse);
  static_assert(position_range * large_biomes_pos_multiplier % (samples_square_size * 2 * 4) == 0);
  
  static constexpr uint32_t POSITION_STEP = position_range * large_biomes_pos_multiplier / 4 / samples_square_size;
  static constexpr int32_t POSITION_OFFSET = -(int32_t)(POSITION_STEP * (samples_square_size - 1) / 2);

  static constexpr uint32_t THREADS_PER_INPUT = std::min(samples, THREADS_PER_BLOCK);
  static constexpr uint32_t LOOPS_PER_THREAD = samples / THREADS_PER_INPUT;
  static constexpr uint32_t INPUTS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_INPUT;

  static void run(InputBuffer<SeedPosition> inputs, OutputBuffer<SeedPosition> outputs, const NoiseCacheInitializer::NoiseCache* __restrict__ noise_caches, hipStream_t stream);
};

template <typename T>
__global__ __launch_bounds__(T::THREADS_PER_BLOCK) void kernel(InputBuffer<SeedPosition> inputs, OutputBuffer<SeedPosition> outputs, const NoiseCacheInitializer::NoiseCache* __restrict__ noise_caches) {

  const uint32_t input_count = *inputs.len;
  
  constexpr uint32_t THREADS_PER_INPUT = T::THREADS_PER_INPUT;
  constexpr uint32_t INPUTS_PER_BLOCK = T::INPUTS_PER_BLOCK;
  
  constexpr uint32_t shift = compile_time_log2(T::THREADS_PER_INPUT);
  constexpr uint32_t mask = T::THREADS_PER_INPUT - 1;
  const uint32_t block_local_input_idx = threadIdx.x >> shift;
  const uint32_t thread_sample_idx = threadIdx.x & mask;

  constexpr int32_t z_step = (int32_t)(T::POSITION_STEP * (T::samples_square_size / T::LOOPS_PER_THREAD));

  __shared__ uint32_t s_valid_counts[INPUTS_PER_BLOCK];
  __shared__ int32_t s_centroid_sums[INPUTS_PER_BLOCK][2];

  for (uint32_t block_input_base = blockIdx.x * INPUTS_PER_BLOCK; 
       block_input_base < input_count; 
       block_input_base += gridDim.x * INPUTS_PER_BLOCK) {
    
    const uint32_t input_index = block_input_base + block_local_input_idx;
    const bool is_valid_input = (input_index < input_count);

    if (threadIdx.x < T::INPUTS_PER_BLOCK) {
      if constexpr (T::samples > 32) {
        s_valid_counts[threadIdx.x] = 0;
        if constexpr (T::move_center) {
          s_centroid_sums[threadIdx.x][0] = 0;
          s_centroid_sums[threadIdx.x][1] = 0;
        }
      }
    }
    __syncthreads();

    uint32_t total_valid = 0;
    int32_t centroid_offset_x = 0;
    int32_t centroid_offset_z = 0;
    SeedPosition input = {};

    if (is_valid_input) {
      input = inputs.data[input_index];

      const uint32_t x_index = thread_sample_idx % T::samples_square_size;
      uint32_t z_index = thread_sample_idx / T::samples_square_size;
      if constexpr (T::samples_square_sparse) {
        z_index = z_index * 2 + ((x_index & 1) ^ T::flipped_sparse_samples);
      }

      int32_t x = input.x + (int32_t)(x_index * T::POSITION_STEP) + T::POSITION_OFFSET;
      int32_t z = input.z + (int32_t)(z_index * T::POSITION_STEP) + T::POSITION_OFFSET;

      #pragma unroll
      for (uint32_t i = 0; i < T::LOOPS_PER_THREAD; i++) {
        float val;
        if constexpr (T::only_a_octaves) {
          val = noise_caches[input.seed_index].template sample_only_a<T::octaves>(x, 0, z);
        } else {
          val = noise_caches[input.seed_index].template sample<T::octaves>(x, 0, z);
        }

        const bool valid = (val < T::noise_threshold);
        total_valid += reduce_add_32((uint32_t)valid);

        if constexpr (T::move_center) {
          if (valid) {
            centroid_offset_x += x - input.x;
            centroid_offset_z += z - input.z;
          }
        }
        z += z_step;
      }
    }

    if constexpr (T::samples > 32) {
      if (is_valid_input && (threadIdx.x % 32 == 0)) {
        atomicAdd(&s_valid_counts[block_local_input_idx], total_valid);
      }
      __syncthreads();
      if (is_valid_input) {
        total_valid = s_valid_counts[block_local_input_idx];
      }
    }

    if constexpr (T::move_center) {
      if (is_valid_input) {
        centroid_offset_x = reduce_add_32(centroid_offset_x);
        centroid_offset_z = reduce_add_32(centroid_offset_z);
      }
      if constexpr (T::samples > 32) {
        if (is_valid_input && (threadIdx.x % 32 == 0)) {
          atomicAdd(&s_centroid_sums[block_local_input_idx][0], centroid_offset_x);
          atomicAdd(&s_centroid_sums[block_local_input_idx][1], centroid_offset_z);
        }
        __syncthreads();
        if (is_valid_input) {
          centroid_offset_x = s_centroid_sums[block_local_input_idx][0];
          centroid_offset_z = s_centroid_sums[block_local_input_idx][1];
        }
      }
      if (is_valid_input && total_valid != 0) {
        centroid_offset_x /= (int32_t)total_valid;
        centroid_offset_z /= (int32_t)total_valid;
      }
    }

    if (is_valid_input && (total_valid >= T::min_count)) {
      if (thread_sample_idx == 0) {
        uint32_t result_index = atomicAdd(outputs.len, 1);
        if (result_index < outputs.max_len) {
          outputs.data[result_index] = {input.seed_index, input.x + centroid_offset_x, input.z + centroid_offset_z};
        }
      }
    }

    if constexpr (T::samples > 32) {
      __syncthreads();
    }
  }
}

template <int32_t NOISE_THRESHOLD, size_t OCTAVES, uint32_t POSITION_RANGE, uint32_t SAMPLES, uint32_t MIN_COUNT, bool FLIPPED_SPARSE_SAMPLES, bool MOVE_CENTER, bool ONLY_A_OCTAVES>
void RefinementFilterTemplate<NOISE_THRESHOLD, OCTAVES, POSITION_RANGE, SAMPLES, MIN_COUNT, FLIPPED_SPARSE_SAMPLES, MOVE_CENTER, ONLY_A_OCTAVES>::run(InputBuffer<SeedPosition> inputs, OutputBuffer<SeedPosition> outputs, const NoiseCacheInitializer::NoiseCache* __restrict__ noise_caches, hipStream_t stream) {
  using T = RefinementFilterTemplate<NOISE_THRESHOLD, OCTAVES, POSITION_RANGE, SAMPLES, MIN_COUNT, FLIPPED_SPARSE_SAMPLES, MOVE_CENTER, ONLY_A_OCTAVES>;
  kernel<T><<<32 * 256, T::THREADS_PER_BLOCK, 0, stream>>>(inputs, outputs, noise_caches);
  TRY_HIP(hipGetLastError());
  cap_gpu_buffer_length(outputs.len, outputs.max_len, stream);
}
} // namespace MultiOctaveRefinementFilter

template <int32_t MIN_COUNT, bool IS_A>
struct CoarseOctaveFilterConfig {
  static constexpr bool is_a = IS_A;
  static constexpr int32_t min_count = MIN_COUNT;
  static constexpr uint32_t THREADS_PER_BLOCK = 256;
  static constexpr uint32_t WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;

  static constexpr auto &octaves = IS_A ? selected_continentalness_config.octaves_a
                                        : selected_continentalness_config.octaves_b;
  static constexpr float freq_mult_0 = (float)octaves[0].freq_mult;
  static constexpr float freq_mult_1 = (float)octaves[1].freq_mult;
  static constexpr float amp_mult_0 = (float)octaves[0].amp_mult;
  static constexpr float amp_mult_1 = (float)octaves[1].amp_mult;

  using T = MultiOctaveRefinementFilter::RefinementFilterTemplate<-5500, 3, 8 * 1024, 256, MIN_COUNT, false, false, IS_A>;
  static constexpr int32_t POSITION_STEP = T::POSITION_STEP;
  static constexpr int32_t POSITION_OFFSET = T::POSITION_OFFSET;
  static constexpr float noise_threshold = T::noise_threshold;
};

__global__ void copy_buffer_kernel(InputBuffer<SeedPosition> inputs, OutputBuffer<SeedPosition> outputs) {
  uint32_t len = *inputs.len;
  if (len > outputs.max_len) {
    len = outputs.max_len;
  }
  
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *outputs.len = len;
  }

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  for (uint32_t i = tid; i < len; i += stride) {
    outputs.data[i] = inputs.data[i];
  }
}

namespace GridGradientFilter2 {
constexpr uint32_t THREADS_PER_BLOCK = 128;
constexpr uint32_t grid_width = unbound ? 331 : (large_biomes ? 29 : 115); //checking full bounded world / full unbounded
constexpr uint32_t THREADS_PER_SEED = grid_width * grid_width;
constexpr uint32_t grid_half = grid_width / 2;

__global__
__launch_bounds__(THREADS_PER_BLOCK) void kernel(
    InputBuffer<SeedPosition> inputs,
    OutputBuffer<SeedPosition> outputs,
    const NoiseCacheInitializer::NoiseCache* __restrict__ noise_caches)
{
  __shared__ float2 s_kernel_0B[6][6][16];
  for (uint32_t i = threadIdx.x; i < 576; i += blockDim.x) {
    ((float2*)s_kernel_0B)[i] = ((const float2*)c_kernel_0B)[i];
  }
  __syncthreads();

  __shared__ alignas(16) MinecraftImprovedNoise octave_0B;
  __shared__ alignas(16) MinecraftImprovedNoise octave_0A;
  __shared__ alignas(16) MinecraftImprovedNoise octave_1A;
  __shared__ float conv_z0[6][256];
  __shared__ float conv_z1[6][256];
  __shared__ uint32_t s_block_match_count[4];

  constexpr int32_t cell_size_0A = (int32_t)(1.0f / selected_continentalness_config.octaves_a[0].freq_mult) * 256;
  const float freq_mult_b = selected_continentalness_config.octaves_b[0].freq_mult;
  const int32_t grid_half_s = (int32_t)grid_half;

  const uint32_t input_count = *inputs.len;
  for (uint32_t input_index = blockIdx.x; input_index < input_count; input_index += gridDim.x) {
    const SeedPosition input = inputs.data[input_index];

    __syncthreads();

    if (threadIdx.x < 17) {
      reinterpret_cast<uint4*>(&octave_0A)[threadIdx.x] =
          reinterpret_cast<const uint4*>(&noise_caches[input.seed_index].octaves[0])[threadIdx.x];
      reinterpret_cast<uint4*>(&octave_1A)[threadIdx.x] =
          reinterpret_cast<const uint4*>(&noise_caches[input.seed_index].octaves[2])[threadIdx.x];
      reinterpret_cast<uint4*>(&octave_0B)[threadIdx.x] =
          reinterpret_cast<const uint4*>(&noise_caches[input.seed_index].octaves[1])[threadIdx.x];
    }

    __syncthreads();

    uint32_t local_matches = 0;
    const int32_t POSITION_STEP = CoarseOctaveFilterConfig<27, true>::POSITION_STEP;
    const int32_t POSITION_OFFSET = CoarseOctaveFilterConfig<27, true>::POSITION_OFFSET;
    const float noise_a_freq_mult_0 = (float)selected_continentalness_config.octaves_a[0].freq_mult;
    const float noise_a_freq_mult_1 = (float)selected_continentalness_config.octaves_a[1].freq_mult;
    const float noise_a_amp_mult_0 = (float)selected_continentalness_config.octaves_a[0].amp_mult;
    const float noise_a_amp_mult_1 = (float)selected_continentalness_config.octaves_a[1].amp_mult;

    #pragma unroll
    for (uint32_t s = 0; s < 2; ++s) {
      const uint32_t sample_idx = threadIdx.x + s * 128;
      const uint32_t x_idx = sample_idx % 16;
      const uint32_t z_idx = sample_idx / 16;
      
      const int32_t x_world = input.x + (int32_t)(x_idx * POSITION_STEP) + POSITION_OFFSET;
      const int32_t z_world = input.z + (int32_t)(z_idx * POSITION_STEP) + POSITION_OFFSET;

      const float val0 = sample_minecraft_noise(octave_0A, x_world * noise_a_freq_mult_0, 0.0f, z_world * noise_a_freq_mult_0) * noise_a_amp_mult_0;
      const float val1 = sample_minecraft_noise(octave_1A, x_world * noise_a_freq_mult_1, 0.0f, z_world * noise_a_freq_mult_1) * noise_a_amp_mult_1;
      
      if (val0 + val1 < -0.5500f) {
        local_matches++;
      }
    }

    local_matches = reduce_add_32(local_matches);
    if ((threadIdx.x & 31) == 0) {
      s_block_match_count[threadIdx.x >> 5] = local_matches;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
      uint32_t block_sum = (threadIdx.x < 4) ? s_block_match_count[threadIdx.x] : 0;
      block_sum = reduce_add_32(block_sum);
      if (threadIdx.x == 0) {
        s_block_match_count[0] = block_sum;
      }
    }
    __syncthreads();

    if (s_block_match_count[0] < 27) {
      continue;
    }

    for (int32_t noise_z_index = threadIdx.x; noise_z_index < 256; noise_z_index += blockDim.x) {
      compute_z_conv(s_kernel_0B, octave_0B, noise_z_index, conv_z0, conv_z1);
    }

    __syncthreads();

    const float xo_B = octave_0B.xo;
    const int32_t yo_B = (int32_t)octave_0B.yo;
    const float zo_B = octave_0B.zo;

    for (uint32_t grid_col = 0; grid_col < grid_width; ++grid_col) {
      const int32_t grid_offset_x = ((int32_t)grid_col - grid_half_s) * cell_size_0A;
      const int32_t world_x = input.x + grid_offset_x;
      const int32_t grid_coord_x = __float2int_rd(world_x * freq_mult_b + xo_B - 2.0f);

      int32_t precomputed_xy_permutations[2][6];
#pragma unroll
      for (int32_t offset_x = 0; offset_x < 6; ++offset_x) {
        const int32_t idx_x = octave_0B.p[(grid_coord_x + offset_x) & 0xFF];
#pragma unroll
        for (int32_t offset_y = 0; offset_y < 2; ++offset_y) {
          precomputed_xy_permutations[offset_y][offset_x] = octave_0B.p[(idx_x + yo_B + offset_y) & 0xFF];
        }
      }

      for (uint32_t grid_row = threadIdx.x; grid_row < grid_width; grid_row += blockDim.x) {
        const int32_t grid_offset_z = ((int32_t)grid_row - grid_half_s) * cell_size_0A;
        const int32_t world_z = input.z + grid_offset_z;

        const int32_t grid_coord_z = __float2int_rd(world_z * freq_mult_b + zo_B - 2.0f);
        const int32_t noise_z_offset = grid_coord_z & 0xFF;

        int32_t noise_permutations_y0[6];
        int32_t noise_permutations_y1[6];
#pragma unroll
        for (int32_t i = 0; i < 6; ++i) {
          noise_permutations_y0[i] = precomputed_xy_permutations[0][i] + noise_z_offset;
          noise_permutations_y1[i] = precomputed_xy_permutations[1][i] + noise_z_offset;
        }

        const float gate = eval_conv_center_2x2(conv_z0, conv_z1, noise_permutations_y0, noise_permutations_y1);
        if (gate >= GRADIENT_VECTORS_2_PREFILTER_THRESHOLD) {
          const float score = gate + eval_conv_edges_12(conv_z0, conv_z1, noise_permutations_y0, noise_permutations_y1);
          if (score > GRADIENT_VECTORS_2_FINAL_THRESHOLD) {
            uint32_t result_index = atomicAdd(outputs.len, 1);
            if (result_index < outputs.max_len) {
              outputs.data[result_index] = {input.seed_index, world_x, world_z};
            }
          }
        }
      }
    }
  }
}

void run(
    const InputBuffer<SeedPosition> inputs,
    OutputBuffer<SeedPosition> outputs,
    const NoiseCacheInitializer::NoiseCache* __restrict__ noise_caches,
    hipStream_t stream)
{
  kernel<<<2048, THREADS_PER_BLOCK, 0, stream>>>(inputs, outputs, noise_caches);
  TRY_HIP(hipGetLastError());
  cap_gpu_buffer_length(outputs.len, outputs.max_len, stream);
}
} // namespace GridGradientFilter2

template <typename C>
__global__ __launch_bounds__(C::THREADS_PER_BLOCK) void coarse_octave_filter_kernel(
    InputBuffer<SeedPosition> inputs,
    OutputBuffer<SeedPosition> outputs,
    const NoiseCacheInitializer::NoiseCache* __restrict__ noise_caches)
{
  __shared__ MinecraftImprovedNoise s_noise_oct0[C::WARPS_PER_BLOCK];
  __shared__ MinecraftImprovedNoise s_noise_oct1[C::WARPS_PER_BLOCK];

  const uint32_t lane_id = threadIdx.x & 31u;
  const uint32_t warp_block_idx = threadIdx.x >> 5;
  const uint32_t warp_global_idx = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const uint32_t total_warps = (gridDim.x * blockDim.x) >> 5;

  MinecraftImprovedNoise &noise_oct0 = s_noise_oct0[warp_block_idx];
  MinecraftImprovedNoise &noise_oct1 = s_noise_oct1[warp_block_idx];

  const uint32_t z_sample_idx = lane_id >> 1;
  const uint32_t x_start_sample_idx = (lane_id & 1u) * 8u;

  constexpr uint32_t uint4_words_per_noise = sizeof(MinecraftImprovedNoise) / sizeof(uint4);
  static_assert(sizeof(MinecraftImprovedNoise) % sizeof(uint4) == 0);

  const uint32_t input_count = *inputs.len;
  for (uint32_t input_index = warp_global_idx; input_index < input_count; input_index += total_warps) {
    const SeedPosition input = inputs.data[input_index];
    const uint32_t seed_index = input.seed_index;

    {
      const uint4 *source_ptr_oct0;
      const uint4 *source_ptr_oct1;
      if constexpr (C::is_a) {
        source_ptr_oct0 = reinterpret_cast<const uint4 *>(&noise_caches[seed_index].octaves[0]);
        source_ptr_oct1 = reinterpret_cast<const uint4 *>(&noise_caches[seed_index].octaves[2]);
      } else {
        source_ptr_oct0 = reinterpret_cast<const uint4 *>(&noise_caches[seed_index].octaves[1]);
        source_ptr_oct1 = reinterpret_cast<const uint4 *>(&noise_caches[seed_index].octaves[3]);
      }
      uint4 *dest_ptr_oct0 = reinterpret_cast<uint4 *>(&noise_oct0);
      uint4 *dest_ptr_oct1 = reinterpret_cast<uint4 *>(&noise_oct1);
      if (lane_id < uint4_words_per_noise) {
        dest_ptr_oct0[lane_id] = source_ptr_oct0[lane_id];
        dest_ptr_oct1[lane_id] = source_ptr_oct1[lane_id];
      }
    }
    __syncwarp();

    const float octave0_xo = noise_oct0.xo;
    const float octave0_yo = noise_oct0.yo;
    const float octave0_zo = noise_oct0.zo;
    const float octave1_xo = noise_oct1.xo;
    const float octave1_yo = noise_oct1.yo;
    const float octave1_zo = noise_oct1.zo;

    const int32_t z_world = input.z + (int32_t)(z_sample_idx * C::POSITION_STEP) + C::POSITION_OFFSET;
    const int32_t x_base = input.x + (int32_t)(x_start_sample_idx * C::POSITION_STEP) + C::POSITION_OFFSET;

    int32_t grid_y0, grid_z0, grid_y1, grid_z1;
    float frac_y0, frac_z0, frac_y1, frac_z1;
    float fade_y0, fade_z0, fade_y1, fade_z1;

    compute_perlin_coords(octave0_yo, grid_y0, frac_y0, fade_y0);
    compute_perlin_coords(z_world * C::freq_mult_0 + octave0_zo, grid_z0, frac_z0, fade_z0);

    compute_perlin_coords(octave1_yo, grid_y1, frac_y1, fade_y1);
    compute_perlin_coords(z_world * C::freq_mult_1 + octave1_zo, grid_z1, frac_z1, fade_z1);

    int32_t last_grid_x0 = 0;
    int32_t last_grid_x1 = 0;
    bool is_x0_cached = false;
    bool is_x1_cached = false;
    
    uint8_t octave0_corner_000, octave0_corner_100, octave0_corner_010, octave0_corner_110;
    uint8_t octave0_corner_001, octave0_corner_101, octave0_corner_011, octave0_corner_111;
    
    uint8_t octave1_corner_000, octave1_corner_100, octave1_corner_010, octave1_corner_110;
    uint8_t octave1_corner_001, octave1_corner_101, octave1_corner_011, octave1_corner_111;

    uint32_t local_matches = 0;
#pragma unroll
    for (uint32_t sample_offset = 0; sample_offset < 8; sample_offset++) {
      const int32_t x_world = x_base + (int32_t)(sample_offset * C::POSITION_STEP);

      int32_t grid_x0;
      float frac_x0, fade_x0;
      compute_perlin_coords(x_world * C::freq_mult_0 + octave0_xo, grid_x0, frac_x0, fade_x0);
      
      if (!is_x0_cached || grid_x0 != last_grid_x0) {
        compute_cell(noise_oct0, grid_x0, grid_y0, grid_z0, 
                     octave0_corner_000, octave0_corner_100, octave0_corner_010, octave0_corner_110, 
                     octave0_corner_001, octave0_corner_101, octave0_corner_011, octave0_corner_111);
        last_grid_x0 = grid_x0;
        is_x0_cached = true;
      }
      const float noise_val_octave0 = trilinear_interp(frac_x0, frac_y0, frac_z0, fade_x0, fade_y0, fade_z0, 
                                             octave0_corner_000, octave0_corner_100, octave0_corner_010, octave0_corner_110, 
                                             octave0_corner_001, octave0_corner_101, octave0_corner_011, octave0_corner_111);

      int32_t grid_x1;
      float frac_x1, fade_x1;
      compute_perlin_coords(x_world * C::freq_mult_1 + octave1_xo, grid_x1, frac_x1, fade_x1);

      if (!is_x1_cached || grid_x1 != last_grid_x1) {
        compute_cell(noise_oct1, grid_x1, grid_y1, grid_z1, 
                     octave1_corner_000, octave1_corner_100, octave1_corner_010, octave1_corner_110, 
                     octave1_corner_001, octave1_corner_101, octave1_corner_011, octave1_corner_111);
        last_grid_x1 = grid_x1;
        is_x1_cached = true;
      }
      const float noise_val_octave1 = trilinear_interp(frac_x1, frac_y1, frac_z1, fade_x1, fade_y1, fade_z1, 
                                             octave1_corner_000, octave1_corner_100, octave1_corner_010, octave1_corner_110, 
                                             octave1_corner_001, octave1_corner_101, octave1_corner_011, octave1_corner_111);

      float combined_noise_val = noise_val_octave0 * C::amp_mult_0 + noise_val_octave1 * C::amp_mult_1;
      local_matches += (combined_noise_val < C::noise_threshold) ? 1u : 0u;
    }

    const uint32_t total_warp_matches = reduce_add_32(local_matches);
    if (lane_id == 0 && total_warp_matches >= C::min_count) {
      uint32_t result_index = atomicAdd(outputs.len, 1);
      if (result_index < outputs.max_len) {
        outputs.data[result_index] = {seed_index, input.x, input.z};
      }
    }
    __syncwarp();
  }
}

// cactus was here :)
namespace CoarseOctaveFilter {
using T = MultiOctaveRefinementFilter::RefinementFilterTemplate<-5500, 3, 8 * 1024, 256, 20, false, false, false>;
using Config = CoarseOctaveFilterConfig<20, false>;

static_assert(T::samples == 256);
static_assert(T::samples_square_size == 16);
static_assert(!T::samples_square_sparse);
static_assert(T::octaves == 3 && !T::only_a_octaves);
static_assert(!T::move_center);

void run(InputBuffer<SeedPosition> inputs, OutputBuffer<SeedPosition> outputs, const NoiseCacheInitializer::NoiseCache* __restrict__ noise_caches, hipStream_t stream) {
  coarse_octave_filter_kernel<Config><<<32 * 256, Config::THREADS_PER_BLOCK, 0, stream>>>(inputs, outputs, noise_caches);
  TRY_HIP(hipGetLastError());
  cap_gpu_buffer_length(outputs.len, outputs.max_len, stream);
}
} // namespace CoarseOctaveFilter

namespace DeduplicateFilter {
__global__ void kernel(InputBuffer<SeedPosition> inputs, OutputBuffer<SeedPosition> outputs, uint32_t *flags) {
  uint32_t len = *inputs.len;
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len; idx += gridDim.x * blockDim.x) {
    SeedPosition pos = inputs.data[idx];
    if (atomicExch(&flags[pos.seed_index], 1) == 0) {
      uint32_t out_idx = atomicAdd(outputs.len, 1);
      if (out_idx < outputs.max_len) {
        outputs.data[out_idx] = pos;
      }
    }
  }
}

void run(InputBuffer<SeedPosition> inputs, OutputBuffer<SeedPosition> outputs, uint32_t *flags, hipStream_t stream) {
  TRY_HIP(hipMemsetAsync(flags, 0, NoiseCacheInitializer::MAX_CACHE_SEEDS * sizeof(uint32_t), stream));
  kernel<<<256, 256, 0, stream>>>(inputs, outputs, flags);
  TRY_HIP(hipGetLastError());
  cap_gpu_buffer_length(outputs.len, outputs.max_len, stream);
}
} // namespace DeduplicateFilter

using Refine12 = MultiOctaveRefinementFilter::RefinementFilterTemplate<-10500, 12, 8 * 1024, 256, 24, false, true, false>;
using Refine14 = MultiOctaveRefinementFilter::RefinementFilterTemplate<-10500, 14, 8 * 1024, 1024, 110, false, true, false>;
using Refine16 = MultiOctaveRefinementFilter::RefinementFilterTemplate<-10500, 16, 10 * 1024, 4096, 340, false, false, false>;
using Refine18 = MultiOctaveRefinementFilter::RefinementFilterTemplate<-10500, 18, 10 * 1024, 16384, 1540, false, false, false>; // zajonc was here :D, use 1600 instead of 1540 because of colab shitty cpus

GpuThread::GpuThread(int device, SeedIterator &input, GpuOutputs &outputs)
    : Thread(), device(device), input(input), outputs(outputs) {
  start();
}

void GpuThread::run() {
  std::printf("Initializing device %d\n", device);

  TRY_HIP(hipSetDevice(device));
  TRY_HIP(hipFuncSetAttribute(reinterpret_cast<const void*>(GridGradientFilter1::kernel), hipFuncAttributePreferredSharedMemoryCarveout, 25));
  initialize_convolution_kernels();

  hipStream_t stream;
  TRY_HIP(hipStreamCreate(&stream));

  BufferLengths host_buffer_lens;
  BufferLengths *device_buffer_lens;
  TRY_HIP(hipMalloc(&device_buffer_lens, sizeof(*device_buffer_lens)));

  std::printf("Running device %d\n\n", device);

  DeviceBuffer prefiltered_seeds_buffer(sizeof(uint64_t) * NoiseCacheInitializer::MAX_CACHE_SEEDS);
  DeviceBuffer noise_cache_buffer(sizeof(NoiseCacheInitializer::NoiseCache) * NoiseCacheInitializer::MAX_CACHE_SEEDS);
  DeviceBuffer flags_buffer(sizeof(uint32_t) * NoiseCacheInitializer::MAX_CACHE_SEEDS);
  DeviceBuffer compact_seeds_buffer(sizeof(uint32_t) * NoiseCacheInitializer::MAX_CACHE_SEEDS);
  DeviceBuffer ping_pong_buffer_1(UINT32_C(1) << 31);
  DeviceBuffer ping_pong_buffer_2(UINT32_C(1) << 29);
  
  constexpr uint32_t MAX_BATCH_SEEDS = 1 << 20;
  DeviceBuffer device_gathered_seeds_buffer(sizeof(uint64_t) * MAX_BATCH_SEEDS);
  std::vector<SeedPosition> host_seed_positions;
  std::vector<uint64_t> host_gathered_seeds;
  
  std::vector<BaseStage*> pipeline;
  pipeline.reserve(16);

  NoiseCacheInitializer::NoiseCache *noise_cache_array = (NoiseCacheInitializer::NoiseCache *)noise_cache_buffer.data;
  HipEventWrapper event_start;

  Stage<uint64_t> stage_prefilter_seeds(pipeline, "Prefilter Seeds", prefiltered_seeds_buffer, &device_buffer_lens->prefilter_seeds_count, &host_buffer_lens.prefilter_seeds_count, nullptr, SeedPreFilter::SEED_BATCH_SIZE);
  
  CustomStage stage_init_noise_cache_coarse(pipeline, "Init Noise Cache Coarse", &host_buffer_lens.prefilter_seeds_count, &host_buffer_lens.prefilter_seeds_count, 1, NoiseCacheInitializer::MAX_CACHE_SEEDS);

  Stage<SeedPosition> stage_gradient_vectors_1(pipeline, "Gradient Vectors 1", ping_pong_buffer_2, &device_buffer_lens->gradient_vectors_1_count, &host_buffer_lens.gradient_vectors_1_count, &host_buffer_lens.prefilter_seeds_count, 256 * 256);

  Stage<SeedPosition> stage_gradient_vectors_2(pipeline, "Gradient Vectors 2", ping_pong_buffer_1, &device_buffer_lens->gradient_vectors_2_count, &host_buffer_lens.gradient_vectors_2_count, &host_buffer_lens.gradient_vectors_1_count, GridGradientFilter2::THREADS_PER_SEED);

  Stage<SeedPosition> stage_coarse_octave_b(pipeline, "Coarse Octave", ping_pong_buffer_2, &device_buffer_lens->coarse_octave_b_count, &host_buffer_lens.coarse_octave_b_count, &host_buffer_lens.gradient_vectors_2_count, 1);

  CustomStage stage_init_noise_cache_fine(pipeline, "Init Noise Cache Fine", &host_buffer_lens.coarse_octave_b_count, &host_buffer_lens.fine_seeds_count, 1, NoiseCacheInitializer::MAX_CACHE_SEEDS);

  Stage<SeedPosition> stage_refinement_12_octaves(pipeline, "Refinement (12 Octaves)", ping_pong_buffer_1, &device_buffer_lens->refinement_counts[0], &host_buffer_lens.refinement_counts[0], &host_buffer_lens.coarse_octave_b_count, 1);
  
  Stage<SeedPosition> stage_refinement_14_octaves(pipeline, "Refinement (14 Octaves)", ping_pong_buffer_2, &device_buffer_lens->refinement_counts[1], &host_buffer_lens.refinement_counts[1], &host_buffer_lens.refinement_counts[0], 1);
  
  Stage<SeedPosition> stage_refinement_16_octaves(pipeline, "Refinement (16 Octaves)", ping_pong_buffer_1, &device_buffer_lens->refinement_counts[2], &host_buffer_lens.refinement_counts[2], &host_buffer_lens.refinement_counts[1], 1);
  
  Stage<SeedPosition> stage_refinement_18_octaves(pipeline, "Refinement (18 Octaves)", ping_pong_buffer_2, &device_buffer_lens->refinement_counts[3], &host_buffer_lens.refinement_counts[3], &host_buffer_lens.refinement_counts[2], 1);

  Stage<SeedPosition> stage_dedup(pipeline, "Deduplicate", ping_pong_buffer_1, &device_buffer_lens->refinement_counts[4], &host_buffer_lens.refinement_counts[4], &host_buffer_lens.refinement_counts[3], 1);

  auto start = std::chrono::steady_clock::now();

  for (uint32_t i = 0; !should_stop(); i++) { // :P simpleased here o/
    uint64_t start_seed = input.next(SeedPreFilter::SEED_BATCH_SIZE);
    
    if (input.stop_seed.has_value() && start_seed >= input.stop_seed.value()) {
        break;
    }

    uint32_t current_count = SeedPreFilter::SEED_BATCH_SIZE;
    if (input.stop_seed.has_value() && start_seed + current_count > input.stop_seed.value()) {
        current_count = input.stop_seed.value() - start_seed;
    }

    TRY_HIP(hipMemsetAsync(device_buffer_lens, 0, sizeof(*device_buffer_lens), stream));

    event_start.record(stream);

    SeedPreFilter::run(start_seed, current_count, stage_prefilter_seeds.output_buffer, stream);
    stage_prefilter_seeds.record(stream);

    NoiseCacheInitializer::run_coarse(stage_prefilter_seeds.output_buffer, noise_cache_array, stream);
    stage_init_noise_cache_coarse.record(stream);

    GridGradientFilter1::run(stage_prefilter_seeds.output_buffer, stage_gradient_vectors_1.output_buffer, noise_cache_array, stream);
    stage_gradient_vectors_1.record(stream);

    GridGradientFilter2::run(stage_gradient_vectors_1.output_buffer, stage_gradient_vectors_2.output_buffer, noise_cache_array, stream);
    stage_gradient_vectors_2.record(stream);

    CoarseOctaveFilter::run(stage_gradient_vectors_2.output_buffer, stage_coarse_octave_b.output_buffer, noise_cache_array, stream);
    stage_coarse_octave_b.record(stream);

    NoiseCacheInitializer::run_fine(stage_prefilter_seeds.output_buffer, noise_cache_array, stage_coarse_octave_b.output_buffer, (uint32_t*)flags_buffer.data, (uint32_t*)compact_seeds_buffer.data, &device_buffer_lens->fine_seeds_count, stream);
    stage_init_noise_cache_fine.record(stream);

    Refine12::run(stage_coarse_octave_b.output_buffer, stage_refinement_12_octaves.output_buffer, noise_cache_array, stream);
    stage_refinement_12_octaves.record(stream);

    Refine14::run(stage_refinement_12_octaves.output_buffer, stage_refinement_14_octaves.output_buffer, noise_cache_array, stream);
    stage_refinement_14_octaves.record(stream);

    Refine16::run(stage_refinement_14_octaves.output_buffer, stage_refinement_16_octaves.output_buffer, noise_cache_array, stream);
    stage_refinement_16_octaves.record(stream);

    Refine18::run(stage_refinement_16_octaves.output_buffer, stage_refinement_18_octaves.output_buffer, noise_cache_array, stream);
    stage_refinement_18_octaves.record(stream);

    DeduplicateFilter::run(stage_refinement_18_octaves.output_buffer, stage_dedup.output_buffer, (uint32_t*)flags_buffer.data, stream);
    stage_dedup.record(stream);

    TRY_HIP(hipMemcpyAsync(&host_buffer_lens, device_buffer_lens, sizeof(host_buffer_lens), hipMemcpyDeviceToHost, stream));
    TRY_HIP(hipStreamSynchronize(stream));

    for (auto* stage : pipeline) {
      stage->cap_len();
    }

    {
      HipEventWrapper *prev_event = &event_start;
      for (auto* stage : pipeline) {
        stage->update(*prev_event);
        prev_event = &stage->event;
      }
    }

    const auto &final_outputs = stage_dedup.output_buffer;
    const auto &final_outputs_len = *stage_dedup.host_output_count;
    if (final_outputs_len > 0) {
      uint32_t len = final_outputs_len;
      host_seed_positions.resize(len);
      host_gathered_seeds.resize(len);

      TRY_HIP(hipMemcpy(host_seed_positions.data(), final_outputs.data, sizeof(SeedPosition) * len, hipMemcpyDeviceToHost));

      if (len <= MAX_BATCH_SEEDS) {
        uint32_t blocks = (len + GATHER_THREADS_PER_BLOCK - 1) / GATHER_THREADS_PER_BLOCK;
        gather_winning_seeds_kernel<<<blocks, GATHER_THREADS_PER_BLOCK, 0, stream>>>(
            stage_prefilter_seeds.output_buffer.data, final_outputs.data,
            (uint64_t*)device_gathered_seeds_buffer.data, len);
        TRY_HIP(hipGetLastError());
        TRY_HIP(hipMemcpy(host_gathered_seeds.data(), device_gathered_seeds_buffer.data,
                            sizeof(uint64_t) * len, hipMemcpyDeviceToHost));
      } else {
        for (uint32_t j = 0; j < len; j++) {
          TRY_HIP(hipMemcpy(&host_gathered_seeds[j],
                              &stage_prefilter_seeds.output_buffer.data[host_seed_positions[j].seed_index],
                              sizeof(uint64_t), hipMemcpyDeviceToHost));
        }
      }

      {
        std::lock_guard lock(outputs.mutex);
        for (uint32_t j = 0; j < len; j++) {
          outputs.queue.push({host_gathered_seeds[j], host_seed_positions[j].x * 4, host_seed_positions[j].z * 4});
        }
      }
    }

    if ((i + 1) % PRINT_INTERVAL == 0) {
      auto end = std::chrono::steady_clock::now();
      double host_total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;

      std::printf("\n");
      std::printf("start_seed = %" PRIi64 "\n", start_seed);

      double kernel_total_time = 0;
      for (auto* stage : pipeline) {
        uint64_t scaled_total_inputs = stage->total_inputs * stage->input_multiplier;
        auto [scaled_input_speed, input_speed_unit] = scale_si(scaled_total_inputs / stage->total_time);
        auto [scaled_output_speed, output_speed_unit] = scale_si(stage->total_outputs / stage->total_time);
        std::printf("%-25s - %9.3f ms | %7.3f %% | %12" PRIu64 " -> %12" PRIu64
                    " | 1 in %11.3f | %7.3f %cips | %7.3f %cops\n",
                    stage->name.c_str(), stage->total_time * 1e3,
                    stage->total_time / host_total_time * 100.0,
                    scaled_total_inputs, stage->total_outputs,
                    (double)scaled_total_inputs / stage->total_outputs,
                    scaled_input_speed, input_speed_unit, scaled_output_speed,
                    output_speed_unit);
        kernel_total_time += stage->total_time;
      }

      uint64_t total_inputs = stage_prefilter_seeds.total_inputs * stage_prefilter_seeds.input_multiplier;
      uint64_t total_outputs = stage_dedup.total_outputs;
      auto [scaled_input_speed, input_speed_unit] = scale_si(total_inputs / host_total_time);
      auto [scaled_output_speed, output_speed_unit] = scale_si(total_outputs / host_total_time);
      std::printf(
          "%-25s - %9.3f ms | %7.3f %% | %12" PRIu64
          " -> %12" PRIu64 " |                  | %7.3f %cips | %7.3f %cops\n",
          "total", host_total_time * 1e3, kernel_total_time / host_total_time * 100.0,
          total_inputs, total_outputs, scaled_input_speed, input_speed_unit,
          scaled_output_speed, output_speed_unit);

      size_t gpu_outputs_size;
      {
        std::lock_guard lock(outputs.mutex);
        gpu_outputs_size = outputs.queue.size();
      }
      std::printf("gpu_outputs.size() = %zu\n", gpu_outputs_size);

      for (auto* stage : pipeline) {
        stage->reset();
      }
      start = end;
    }
  }

  TRY_HIP(hipStreamDestroy(stream));
  TRY_HIP(hipFree(device_buffer_lens));
}