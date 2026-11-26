// Custom CUDA kernels of the pipeline. Every launcher only enqueues work on
// |stream| and never synchronizes with the host.

#ifndef SRC_KERNELS_H_
#define SRC_KERNELS_H_

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdint>

namespace audio {

// Indices into the small per-lane device array of scalar results.
enum DeviceStat {
  kStatMean = 0,  // Written by nppsMean_32f.
  kStatMin = 1,   // Written by nppsMinMax_32f.
  kStatMax = 2,
  kStatDbMax = 3,     // Written by nppsMax_32f on the dB spectrogram.
  kStatPeakPower = 4,  // Written by nppsMaxIndx_32f on the mean spectrum.
  kNumStats = 5,
};

// x = (x - mean) / peak, where mean/min/max come from |stats| (device).
void LaunchNormalize(float* signal, int num_samples, const float* stats,
                     cudaStream_t stream);

// Cuts the signal into Hann-windowed frames: frames[f * frame_length + i].
// Frames f >= num_frames (padding up to padded_frames) are set to zero.
void LaunchFrameAndWindow(const float* signal, int num_samples,
                          int frame_length, int hop, int num_frames,
                          int padded_frames, float* frames,
                          cudaStream_t stream);

// power = |X|^2 * power_scale and db = 10*log10(power + tiny).
void LaunchPowerSpectrum(const cufftComplex* spectrum, int count,
                         float power_scale, float* power, float* db,
                         cudaStream_t stream);

// One thread block per frame: RMS, zero-crossing rate, spectral centroid
// and spectral flux, written feature-major into |features|.
void LaunchFrameFeatures(const float* signal, int num_samples,
                         const float* power, int num_frames, int frame_length,
                         int hop, int num_bins, float bin_hz, float* features,
                         cudaStream_t stream);

// mean_spectrum[k] = average of power[t][k] over all frames t.
void LaunchMeanSpectrum(const float* power, int num_frames, int num_bins,
                        float* mean_spectrum, cudaStream_t stream);

// Maps the dB spectrogram to an 8-bit image covering
// [max_db - range_db, max_db]; low frequencies end up at the bottom.
void LaunchSpectrogramImage(const float* db, int num_frames, int num_bins,
                            const float* stats, float range_db,
                            uint8_t* image, cudaStream_t stream);

}  // namespace audio

#endif  // SRC_KERNELS_H_
