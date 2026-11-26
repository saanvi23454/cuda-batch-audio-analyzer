// GPU batch analyzer: NPP signal functions + batched cuFFT + custom kernels,
// with several CUDA streams ("lanes") so different signals overlap.
//
// This header has no CUDA types, so plain C++ files can include it.

#ifndef SRC_GPU_ANALYZER_H_
#define SRC_GPU_ANALYZER_H_

#include <memory>
#include <string>
#include <vector>

#include "signal_types.h"

namespace audio {

class GpuAnalyzer {
 public:
  // Allocates every buffer, stream, cuFFT plan and NPP scratch area up
  // front for signals of up to |max_samples| samples, so the batch loop
  // itself never allocates (allocation would synchronize the streams).
  GpuAnalyzer(const StftConfig& config, size_t max_samples, int num_streams);
  ~GpuAnalyzer();

  GpuAnalyzer(const GpuAnalyzer&) = delete;
  GpuAnalyzer& operator=(const GpuAnalyzer&) = delete;

  // Analyzes all |signals|. If |want_image[i]| is true the 8-bit spectrogram
  // of signal i is also returned. |results| gets one entry per signal.
  void AnalyzeBatch(const std::vector<Signal>& signals,
                    const std::vector<bool>& want_image,
                    std::vector<SignalAnalysis>* results);

  // GPU name, compute capability and library versions, for the log.
  std::string DeviceDescription() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace audio

#endif  // SRC_GPU_ANALYZER_H_
