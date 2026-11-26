#include "features.h"

#include <algorithm>
#include <cmath>

namespace audio {
namespace {

// Onset picking parameters (tuned on the synthetic dataset with the CPU
// reference, see README). A frame is an onset when its flux is a local
// maximum and exceeds the moving median by a fraction of the largest flux.
// The moving median removes the steady flux that background noise adds.
constexpr double kOnsetRelativeThreshold = 0.25;  // Of the largest flux.
constexpr double kOnsetMedianSeconds = 0.2;       // Half-width of the median.
constexpr double kOnsetPeakRadiusSeconds = 0.03;  // Local-maximum window.
constexpr double kOnsetMinGapSeconds = 0.10;      // Between two onsets.

// Verification tolerances.
constexpr double kMaxRelativeFeatureError = 1e-3;
constexpr double kMaxSpectrumError = 1e-3;
constexpr double kPeakTieTolerance = 1e-4;

// Largest |a - b| over the arrays, divided by the largest |b|.
double RelativeError(const float* a, const float* b, int count) {
  double max_diff = 0.0;
  double max_ref = 0.0;
  for (int i = 0; i < count; ++i) {
    const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    max_diff = std::max(max_diff, std::fabs(diff));
    max_ref = std::max(max_ref, std::fabs(static_cast<double>(b[i])));
  }
  return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

double AbsoluteError(const float* a, const float* b, int count) {
  double max_diff = 0.0;
  for (int i = 0; i < count; ++i) {
    const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    max_diff = std::max(max_diff, std::fabs(diff));
  }
  return max_diff;
}

}  // namespace

std::vector<int> DetectOnsets(const float* flux, int num_frames,
                              const StftConfig& config, int sample_rate) {
  std::vector<int> onsets;
  if (num_frames < 3) return onsets;
  const float max_flux = *std::max_element(flux, flux + num_frames);
  if (max_flux <= 0.0f) return onsets;

  const double frames_per_second =
      static_cast<double>(sample_rate) / config.hop;
  const int radius = std::max(
      1, static_cast<int>(std::lround(kOnsetPeakRadiusSeconds *
                                      frames_per_second)));
  const int min_gap = std::max(
      1, static_cast<int>(std::lround(kOnsetMinGapSeconds *
                                      frames_per_second)));
  const int median_radius = std::max(
      1, static_cast<int>(std::lround(kOnsetMedianSeconds *
                                      frames_per_second)));
  const double margin = kOnsetRelativeThreshold * max_flux;

  std::vector<float> window;
  int last = -min_gap;
  for (int t = 1; t < num_frames; ++t) {
    if (flux[t] < margin) continue;  // Cheap early out.
    const int median_lo = std::max(0, t - median_radius);
    const int median_hi = std::min(num_frames - 1, t + median_radius);
    window.assign(flux + median_lo, flux + median_hi + 1);
    std::nth_element(window.begin(), window.begin() + window.size() / 2,
                     window.end());
    if (flux[t] < window[window.size() / 2] + margin) continue;
    const int lo = std::max(0, t - radius);
    const int hi = std::min(num_frames - 1, t + radius);
    const float local_max = *std::max_element(flux + lo, flux + hi + 1);
    if (flux[t] < local_max) continue;
    if (t - last < min_gap) continue;
    onsets.push_back(t);
    last = t;
  }
  return onsets;
}

double DominantFrequencyHz(const std::vector<float>& mean_spectrum,
                           int peak_bin, const StftConfig& config,
                           int sample_rate) {
  const int bins = static_cast<int>(mean_spectrum.size());
  double offset = 0.0;
  if (peak_bin > 0 && peak_bin + 1 < bins) {
    const double floor = 1e-30;
    const double a = std::log(mean_spectrum[peak_bin - 1] + floor);
    const double b = std::log(mean_spectrum[peak_bin] + floor);
    const double c = std::log(mean_spectrum[peak_bin + 1] + floor);
    const double denominator = a - 2.0 * b + c;
    if (denominator < 0.0) {
      offset = std::max(-0.5, std::min(0.5, 0.5 * (a - c) / denominator));
    }
  }
  return (peak_bin + offset) * config.BinHz(sample_rate);
}

double MeanOf(const float* values, int count) {
  if (count <= 0) return 0.0;
  double sum = 0.0;
  for (int i = 0; i < count; ++i) sum += values[i];
  return sum / count;
}

Comparison CompareAnalyses(const SignalAnalysis& gpu,
                           const SignalAnalysis& cpu,
                           const StftConfig& config) {
  Comparison result;
  if (gpu.num_frames != cpu.num_frames ||
      gpu.mean_spectrum.size() != cpu.mean_spectrum.size() ||
      gpu.features.size() != cpu.features.size()) {
    return result;  // passed == false.
  }
  const int frames = cpu.num_frames;

  // A different peak bin is fine only when both bins are equally strong.
  const float ref_at_gpu_peak = cpu.mean_spectrum[gpu.peak_bin];
  const float ref_at_cpu_peak = cpu.mean_spectrum[cpu.peak_bin];
  result.same_peak =
      gpu.peak_bin == cpu.peak_bin ||
      std::fabs(ref_at_gpu_peak - ref_at_cpu_peak) <=
          kPeakTieTolerance * ref_at_cpu_peak;

  bool features_ok = true;
  for (int f = 0; f < kNumFeatures; ++f) {
    const FeatureIndex index = static_cast<FeatureIndex>(f);
    if (index == kZeroCrossingRate) {
      // A sample within rounding of zero may land on either side, which
      // changes up to two crossings in a frame.
      result.feature_error[f] =
          AbsoluteError(gpu.Feature(index), cpu.Feature(index), frames);
      features_ok &=
          result.feature_error[f] <= 4.0 / (config.frame_length - 1);
    } else {
      result.feature_error[f] =
          RelativeError(gpu.Feature(index), cpu.Feature(index), frames);
      features_ok &= result.feature_error[f] <= kMaxRelativeFeatureError;
    }
  }
  result.spectrum_error =
      RelativeError(gpu.mean_spectrum.data(), cpu.mean_spectrum.data(),
                    static_cast<int>(cpu.mean_spectrum.size()));
  result.passed = result.same_peak && features_ok &&
                  result.spectrum_error <= kMaxSpectrumError;
  return result;
}

}  // namespace audio
