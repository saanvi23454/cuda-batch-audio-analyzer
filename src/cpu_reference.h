// Single-threaded CPU version of the whole GPU pipeline. It is used to check
// the GPU results and as the baseline for the speed-up numbers.

#ifndef SRC_CPU_REFERENCE_H_
#define SRC_CPU_REFERENCE_H_

#include <complex>
#include <vector>

#include "signal_types.h"

namespace audio {

class CpuAnalyzer {
 public:
  explicit CpuAnalyzer(const StftConfig& config);

  // Same stages and outputs as GpuAnalyzer (except the spectrogram image).
  SignalAnalysis Analyze(const Signal& signal) const;

  // In-place radix-2 FFT of |data| (size == frame_length). Public for tests.
  void Fft(std::vector<std::complex<double>>* data) const;

 private:
  StftConfig config_;
  std::vector<std::complex<double>> twiddles_;  // exp(-2*pi*i*k/N), k < N/2.
  std::vector<int> bit_reverse_;
  std::vector<double> window_;  // Periodic Hann window.
};

}  // namespace audio

#endif  // SRC_CPU_REFERENCE_H_
