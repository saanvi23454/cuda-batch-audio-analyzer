// Host-side post-processing that turns the per-frame arrays produced by the
// GPU (or the CPU reference) into per-signal results, plus the GPU-vs-CPU
// comparison.

#ifndef SRC_FEATURE_UTILS_H_
#define SRC_FEATURE_UTILS_H_

#include <vector>

#include "signal_types.h"

namespace audio {

// Finds note onsets as peaks of the spectral flux curve. Returns frame
// indices in increasing order.
std::vector<int> DetectOnsets(const float* flux, int num_frames,
                              const StftConfig& config, int sample_rate);

// Dominant frequency in Hz: the peak bin refined by fitting a parabola to
// the log power of the bin and its two neighbours.
double DominantFrequencyHz(const std::vector<float>& mean_spectrum,
                           int peak_bin, const StftConfig& config,
                           int sample_rate);

// Plain mean of |count| values (0 for an empty array).
double MeanOf(const float* values, int count);

// How far the GPU results are from the CPU reference.
struct Comparison {
  bool same_peak = false;  // Same peak bin, or a tie within float rounding.
  // Largest difference divided by the largest reference value, per feature.
  // For the zero-crossing rate it is the absolute difference instead.
  double feature_error[kNumFeatures] = {0.0, 0.0, 0.0, 0.0};
  double spectrum_error = 0.0;  // Max |diff| / max reference power.
  bool passed = false;
};

Comparison CompareAnalyses(const SignalAnalysis& gpu,
                           const SignalAnalysis& cpu,
                           const StftConfig& config);

}  // namespace audio

#endif  // SRC_FEATURE_UTILS_H_
