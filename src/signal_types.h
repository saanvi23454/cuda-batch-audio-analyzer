// Data types shared by the GPU pipeline, the CPU reference and the driver.

#ifndef SRC_SIGNAL_TYPES_H_
#define SRC_SIGNAL_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace audio {

// Layout of the short-time Fourier transform (STFT). Both the GPU and the
// CPU paths cut a signal into frames of |frame_length| samples that start
// |hop| samples apart.
struct StftConfig {
  int frame_length = 1024;  // Samples per frame. Must be a power of two.
  int hop = 256;            // Samples between the starts of two frames.

  int NumBins() const { return frame_length / 2 + 1; }

  // Number of frames for a signal of |num_samples| samples. A signal shorter
  // than one frame still gets a single zero-padded frame.
  int NumFrames(size_t num_samples) const {
    if (num_samples <= static_cast<size_t>(frame_length)) return 1;
    return 1 + static_cast<int>((num_samples - frame_length) / hop);
  }

  double BinHz(int sample_rate) const {
    return static_cast<double>(sample_rate) / frame_length;
  }
};

// One mono, 16-bit audio signal.
struct Signal {
  std::string name;  // File name without the directory.
  int sample_rate = 0;
  std::vector<int16_t> samples;
};

// Per-frame features. They are stored feature-major (all RMS values, then all
// zero-crossing rates, ...) so that each feature is one contiguous array.
enum FeatureIndex {
  kRms = 0,               // Root mean square of the normalized samples.
  kZeroCrossingRate = 1,  // Fraction of neighbouring samples that change sign.
  kCentroidHz = 2,        // Power-weighted mean frequency.
  kFlux = 3,              // Positive spectral flux versus the previous frame.
  kNumFeatures = 4,
};

// Everything the pipeline computes for one signal.
struct SignalAnalysis {
  int num_frames = 0;
  std::vector<float> features;       // kNumFeatures * num_frames values.
  std::vector<float> mean_spectrum;  // Mean power of each frequency bin.
  int peak_bin = 0;                  // Strongest bin, ignoring DC (bin 0).
  float gpu_ms = 0.0f;               // GPU latency incl. copies (GPU only).
  // Optional 8-bit spectrogram image: NumBins() rows (high frequencies on
  // top) by num_frames columns. Empty unless it was requested.
  std::vector<uint8_t> spectrogram;

  const float* Feature(FeatureIndex index) const {
    return features.data() + static_cast<size_t>(index) * num_frames;
  }
};

}  // namespace audio

#endif  // SRC_SIGNAL_TYPES_H_
