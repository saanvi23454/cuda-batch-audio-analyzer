#include "cpu_reference.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace audio {
namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

CpuAnalyzer::CpuAnalyzer(const StftConfig& config) : config_(config) {
  const int n = config_.frame_length;
  int log2n = 0;
  while ((1 << log2n) < n) ++log2n;

  bit_reverse_.resize(n);
  for (int i = 0; i < n; ++i) {
    int reversed = 0;
    for (int b = 0; b < log2n; ++b) {
      if (i & (1 << b)) reversed |= 1 << (log2n - 1 - b);
    }
    bit_reverse_[i] = reversed;
  }

  twiddles_.resize(n / 2);
  for (int k = 0; k < n / 2; ++k) {
    twiddles_[k] = std::polar(1.0, -2.0 * kPi * k / n);
  }

  window_.resize(n);
  for (int i = 0; i < n; ++i) {
    window_[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / n);
  }
}

void CpuAnalyzer::Fft(std::vector<std::complex<double>>* data) const {
  std::vector<std::complex<double>>& x = *data;
  const int n = config_.frame_length;
  for (int i = 0; i < n; ++i) {
    if (i < bit_reverse_[i]) std::swap(x[i], x[bit_reverse_[i]]);
  }
  for (int size = 2; size <= n; size <<= 1) {
    const int half = size / 2;
    const int stride = n / size;
    for (int start = 0; start < n; start += size) {
      for (int k = 0; k < half; ++k) {
        const std::complex<double> t =
            twiddles_[k * stride] * x[start + k + half];
        x[start + k + half] = x[start + k] - t;
        x[start + k] += t;
      }
    }
  }
}

SignalAnalysis CpuAnalyzer::Analyze(const Signal& signal) const {
  const int n = static_cast<int>(signal.samples.size());
  const int frame_length = config_.frame_length;
  const int bins = config_.NumBins();
  const int frames = config_.NumFrames(signal.samples.size());
  const double bin_hz = config_.BinHz(signal.sample_rate);

  // Stage 1: remove the DC offset and scale the peak to 1.
  double sum = 0.0;
  double min_value = signal.samples[0];
  double max_value = signal.samples[0];
  for (int16_t s : signal.samples) {
    sum += s;
    min_value = std::min(min_value, static_cast<double>(s));
    max_value = std::max(max_value, static_cast<double>(s));
  }
  const double mean = sum / n;
  const double peak = std::max(max_value - mean, mean - min_value);
  const double scale = peak > 0.0 ? 1.0 / peak : 1.0;
  std::vector<double> x(n);
  for (int i = 0; i < n; ++i) x[i] = (signal.samples[i] - mean) * scale;
  auto sample = [&x, n](int i) { return i < n ? x[i] : 0.0; };

  SignalAnalysis result;
  result.num_frames = frames;
  result.features.assign(static_cast<size_t>(kNumFeatures) * frames, 0.0f);
  result.mean_spectrum.assign(bins, 0.0f);

  std::vector<double> mean_power(bins, 0.0);
  std::vector<double> power(bins, 0.0);
  std::vector<double> previous_power(bins, 0.0);
  std::vector<std::complex<double>> buffer(frame_length);
  const double power_scale = 1.0 / (static_cast<double>(frame_length) *
                                    frame_length);

  for (int t = 0; t < frames; ++t) {
    const int start = t * config_.hop;

    // Stage 2: Hann window + FFT + power spectrum.
    for (int i = 0; i < frame_length; ++i) {
      buffer[i] = std::complex<double>(sample(start + i) * window_[i], 0.0);
    }
    Fft(&buffer);
    for (int k = 0; k < bins; ++k) {
      power[k] = std::norm(buffer[k]) * power_scale;
    }

    // Stage 3: per-frame features.
    double sum_squares = 0.0;
    double crossings = 0.0;
    for (int i = 0; i < frame_length; ++i) {
      const double b = sample(start + i);
      sum_squares += b * b;
      if (i > 0) {
        const double a = sample(start + i - 1);
        if ((a >= 0.0) != (b >= 0.0)) crossings += 1.0;
      }
    }
    double sum_power = 0.0;
    double weighted = 0.0;
    double flux = 0.0;
    for (int k = 0; k < bins; ++k) {
      sum_power += power[k];
      weighted += k * bin_hz * power[k];
      if (t > 0) {
        const double d = std::sqrt(power[k]) - std::sqrt(previous_power[k]);
        if (d > 0.0) flux += d;
      }
      mean_power[k] += power[k];
    }
    result.features[kRms * frames + t] =
        static_cast<float>(std::sqrt(sum_squares / frame_length));
    result.features[kZeroCrossingRate * frames + t] =
        static_cast<float>(crossings / (frame_length - 1));
    result.features[kCentroidHz * frames + t] =
        static_cast<float>(sum_power > 1e-20 ? weighted / sum_power : 0.0);
    result.features[kFlux * frames + t] = static_cast<float>(flux);
    std::swap(power, previous_power);
  }

  // Stage 4: mean spectrum and its strongest bin (DC excluded).
  for (int k = 0; k < bins; ++k) {
    result.mean_spectrum[k] = static_cast<float>(mean_power[k] / frames);
  }
  result.peak_bin = 1;
  for (int k = 2; k < bins; ++k) {
    if (result.mean_spectrum[k] > result.mean_spectrum[result.peak_bin]) {
      result.peak_bin = k;
    }
  }
  return result;
}

}  // namespace audio
