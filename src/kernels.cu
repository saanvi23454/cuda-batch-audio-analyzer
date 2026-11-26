#include "kernels.h"

#include <algorithm>

#include "cuda_check.h"
#include "signal_types.h"

namespace audio {
namespace {

constexpr int kThreads = 256;  // Must be a multiple of the warp size.
constexpr int kMaxGridBlocks = 4096;
constexpr float kTwoPi = 6.283185307179586f;
constexpr float kPowerFloor = 1e-12f;  // Keeps log10 finite in silence.

int GridFor(long long count) {
  const long long blocks = (count + kThreads - 1) / kThreads;
  return static_cast<int>(std::max(1LL, std::min<long long>(blocks,
                                                            kMaxGridBlocks)));
}

__device__ float WarpReduceSum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Sum over the whole block. The result is valid in thread 0. |scratch| needs
// 32 floats of shared memory and can be reused right after the call.
__device__ float BlockReduceSum(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = WarpReduceSum(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  const int num_warps = (blockDim.x + 31) >> 5;
  value = (static_cast<int>(threadIdx.x) < num_warps) ? scratch[lane] : 0.0f;
  if (warp == 0) value = WarpReduceSum(value);
  __syncthreads();
  return value;
}

__global__ void NormalizeKernel(float* x, int n, const float* stats) {
  const float mean = stats[kStatMean];
  const float peak = fmaxf(stats[kStatMax] - mean, mean - stats[kStatMin]);
  const float scale = peak > 0.0f ? 1.0f / peak : 1.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    x[i] = (x[i] - mean) * scale;
  }
}

__global__ void FrameAndWindowKernel(const float* x, int n, int frame_length,
                                     int hop, int num_frames, long long total,
                                     float* frames) {
  for (long long idx = blockIdx.x * static_cast<long long>(blockDim.x) +
                       threadIdx.x;
       idx < total; idx += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int f = static_cast<int>(idx / frame_length);
    const int i = static_cast<int>(idx % frame_length);
    float value = 0.0f;
    if (f < num_frames) {
      const long long s = static_cast<long long>(f) * hop + i;
      if (s < n) {
        const float window =
            0.5f - 0.5f * cosf(kTwoPi * static_cast<float>(i) / frame_length);
        value = x[s] * window;
      }
    }
    frames[idx] = value;
  }
}

__global__ void PowerSpectrumKernel(const cufftComplex* spectrum, int count,
                                    float power_scale, float* power,
                                    float* db) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    const cufftComplex c = spectrum[i];
    const float p = (c.x * c.x + c.y * c.y) * power_scale;
    power[i] = p;
    db[i] = 10.0f * log10f(p + kPowerFloor);
  }
}

// One block per frame. Time-domain features read the normalized signal,
// spectral features read this frame's and the previous frame's power.
__global__ void FrameFeaturesKernel(const float* x, int n, const float* power,
                                    int num_frames, int frame_length, int hop,
                                    int num_bins, float bin_hz,
                                    float* features) {
  __shared__ float scratch[32];
  const int t = blockIdx.x;
  const long long start = static_cast<long long>(t) * hop;

  float sum_squares = 0.0f;
  float crossings = 0.0f;
  for (int i = threadIdx.x; i < frame_length; i += blockDim.x) {
    const long long s = start + i;
    const float b = s < n ? x[s] : 0.0f;
    sum_squares += b * b;
    if (i > 0) {
      const float a = (s - 1) < n ? x[s - 1] : 0.0f;
      if ((a >= 0.0f) != (b >= 0.0f)) crossings += 1.0f;
    }
  }

  const float* p = power + static_cast<size_t>(t) * num_bins;
  const float* previous = t > 0 ? p - num_bins : nullptr;
  float sum_power = 0.0f;
  float weighted = 0.0f;
  float flux = 0.0f;
  for (int k = threadIdx.x; k < num_bins; k += blockDim.x) {
    const float pk = p[k];
    sum_power += pk;
    weighted += static_cast<float>(k) * bin_hz * pk;
    if (previous != nullptr) {
      const float d = sqrtf(pk) - sqrtf(previous[k]);
      if (d > 0.0f) flux += d;
    }
  }

  sum_squares = BlockReduceSum(sum_squares, scratch);
  crossings = BlockReduceSum(crossings, scratch);
  sum_power = BlockReduceSum(sum_power, scratch);
  weighted = BlockReduceSum(weighted, scratch);
  flux = BlockReduceSum(flux, scratch);

  if (threadIdx.x == 0) {
    features[kRms * num_frames + t] = sqrtf(sum_squares / frame_length);
    features[kZeroCrossingRate * num_frames + t] =
        crossings / (frame_length - 1);
    features[kCentroidHz * num_frames + t] =
        sum_power > 1e-20f ? weighted / sum_power : 0.0f;
    features[kFlux * num_frames + t] = flux;
  }
}

// One thread per bin; neighbouring threads read neighbouring bins, so every
// frame row is read with coalesced loads.
__global__ void MeanSpectrumKernel(const float* power, int num_frames,
                                   int num_bins, float* mean_spectrum) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_bins) return;
  float sum = 0.0f;
  for (int t = 0; t < num_frames; ++t) {
    sum += power[static_cast<size_t>(t) * num_bins + k];
  }
  mean_spectrum[k] = sum / num_frames;
}

__global__ void SpectrogramImageKernel(const float* db, int num_frames,
                                       int num_bins, const float* stats,
                                       float range_db, uint8_t* image) {
  const float max_db = stats[kStatDbMax];
  const long long total = static_cast<long long>(num_frames) * num_bins;
  for (long long idx = blockIdx.x * static_cast<long long>(blockDim.x) +
                       threadIdx.x;
       idx < total; idx += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int row = static_cast<int>(idx / num_frames);  // Output row.
    const int t = static_cast<int>(idx % num_frames);
    const int k = num_bins - 1 - row;  // High frequencies at the top.
    const float level =
        (db[static_cast<size_t>(t) * num_bins + k] - (max_db - range_db)) /
        range_db;
    image[idx] = static_cast<uint8_t>(
        255.0f * fminf(1.0f, fmaxf(0.0f, level)) + 0.5f);
  }
}

}  // namespace

void LaunchNormalize(float* signal, int num_samples, const float* stats,
                     cudaStream_t stream) {
  NormalizeKernel<<<GridFor(num_samples), kThreads, 0, stream>>>(
      signal, num_samples, stats);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchFrameAndWindow(const float* signal, int num_samples,
                          int frame_length, int hop, int num_frames,
                          int padded_frames, float* frames,
                          cudaStream_t stream) {
  const long long total = static_cast<long long>(padded_frames) * frame_length;
  FrameAndWindowKernel<<<GridFor(total), kThreads, 0, stream>>>(
      signal, num_samples, frame_length, hop, num_frames, total, frames);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchPowerSpectrum(const cufftComplex* spectrum, int count,
                         float power_scale, float* power, float* db,
                         cudaStream_t stream) {
  PowerSpectrumKernel<<<GridFor(count), kThreads, 0, stream>>>(
      spectrum, count, power_scale, power, db);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchFrameFeatures(const float* signal, int num_samples,
                         const float* power, int num_frames, int frame_length,
                         int hop, int num_bins, float bin_hz, float* features,
                         cudaStream_t stream) {
  FrameFeaturesKernel<<<num_frames, kThreads, 0, stream>>>(
      signal, num_samples, power, num_frames, frame_length, hop, num_bins,
      bin_hz, features);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchMeanSpectrum(const float* power, int num_frames, int num_bins,
                        float* mean_spectrum, cudaStream_t stream) {
  const int blocks = (num_bins + kThreads - 1) / kThreads;
  MeanSpectrumKernel<<<blocks, kThreads, 0, stream>>>(power, num_frames,
                                                       num_bins,
                                                       mean_spectrum);
  CUDA_CHECK(cudaGetLastError());
}

void LaunchSpectrogramImage(const float* db, int num_frames, int num_bins,
                            const float* stats, float range_db,
                            uint8_t* image, cudaStream_t stream) {
  const long long total = static_cast<long long>(num_frames) * num_bins;
  SpectrogramImageKernel<<<GridFor(total), kThreads, 0, stream>>>(
      db, num_frames, num_bins, stats, range_db, image);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace audio
