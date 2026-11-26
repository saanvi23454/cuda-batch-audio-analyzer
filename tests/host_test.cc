// Unit tests for the host-side code. They need no GPU: `make test`.

#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

#include "cli_options.h"
#include "cpu_reference.h"
#include "features.h"
#include "signal_types.h"
#include "wav_io.h"

namespace audio {
namespace {

constexpr double kPi = 3.14159265358979323846;
int g_failures = 0;

void Expect(bool condition, const std::string& what) {
  std::printf("[%s] %s\n", condition ? "ok  " : "FAIL", what.c_str());
  if (!condition) ++g_failures;
}

Signal MakeTone(double hz, double seconds, int rate, double amplitude) {
  Signal s;
  s.name = "tone.wav";
  s.sample_rate = rate;
  const int n = static_cast<int>(seconds * rate);
  s.samples.resize(n);
  for (int i = 0; i < n; ++i) {
    s.samples[i] = static_cast<int16_t>(
        std::lround(amplitude * 32767.0 * std::sin(2.0 * kPi * hz * i / rate)));
  }
  return s;
}

void TestFrameCount() {
  StftConfig c{1024, 256};
  Expect(c.NumFrames(10) == 1, "short signal gets one padded frame");
  Expect(c.NumFrames(1024) == 1, "exactly one frame");
  Expect(c.NumFrames(1024 + 256) == 2, "one hop more gives two frames");
  Expect(c.NumFrames(16000) == 59, "16000 samples -> 59 frames");
  Expect(c.NumBins() == 513, "1024-point FFT has 513 bins");
}

void TestFftMatchesDft() {
  StftConfig c{64, 16};
  CpuAnalyzer cpu(c);
  std::vector<std::complex<double>> data(64);
  for (int i = 0; i < 64; ++i) {
    data[i] = std::complex<double>(std::sin(0.3 * i) + 0.1 * i, 0.0);
  }
  const std::vector<std::complex<double>> input = data;
  cpu.Fft(&data);
  double max_error = 0.0;
  for (int k = 0; k < 64; ++k) {
    std::complex<double> sum(0.0, 0.0);
    for (int i = 0; i < 64; ++i) {
      sum += input[i] * std::polar(1.0, -2.0 * kPi * k * i / 64);
    }
    max_error = std::max(max_error, std::abs(sum - data[k]));
  }
  Expect(max_error < 1e-9, "radix-2 FFT equals the direct DFT");
}

void TestWavRoundTrip() {
  const Signal tone = MakeTone(440.0, 0.25, 8000, 0.5);
  std::string error;
  const std::string path = "host_test_tmp.wav";
  Expect(WriteWav(path, tone, &error), "write WAV");
  Signal back;
  Expect(ReadWav(path, &back, &error), "read WAV back: " + error);
  Expect(back.sample_rate == 8000, "sample rate survives");
  Expect(back.samples == tone.samples, "samples survive");
  std::remove(path.c_str());
}

void TestDominantFrequency() {
  StftConfig c{1024, 256};
  CpuAnalyzer cpu(c);
  const Signal tone = MakeTone(1234.0, 1.0, 16000, 0.8);
  const SignalAnalysis a = cpu.Analyze(tone);
  const double hz = DominantFrequencyHz(a.mean_spectrum, a.peak_bin, c, 16000);
  Expect(std::fabs(hz - 1234.0) < 2.0,
         "1234 Hz tone found at " + std::to_string(hz) + " Hz");
  const double centroid = MeanOf(a.Feature(kCentroidHz), a.num_frames);
  Expect(std::fabs(centroid - 1234.0) < 20.0,
         "centroid of a pure tone is near its frequency");
  const double zcr = MeanOf(a.Feature(kZeroCrossingRate), a.num_frames);
  Expect(std::fabs(zcr - 2.0 * 1234.0 / 16000) < 0.01,
         "zero-crossing rate of a tone is 2 f / rate");
  const double rms = MeanOf(a.Feature(kRms), a.num_frames);
  Expect(std::fabs(rms - 1.0 / std::sqrt(2.0)) < 0.01,
         "RMS of a normalized sine is 1/sqrt(2)");
}

void TestOnsets() {
  // Three notes separated by silence.
  const int rate = 16000;
  Signal s;
  s.sample_rate = rate;
  s.samples.assign(rate * 2, 0);
  const double starts[] = {0.2, 0.8, 1.4};
  for (double start : starts) {
    const int first = static_cast<int>(start * rate);
    for (int i = 0; i < rate * 0.3; ++i) {
      const double env = std::exp(-i / (0.1 * rate));
      s.samples[first + i] = static_cast<int16_t>(
          20000.0 * env * std::sin(2.0 * kPi * 440.0 * i / rate));
    }
  }
  StftConfig c{1024, 256};
  CpuAnalyzer cpu(c);
  const SignalAnalysis a = cpu.Analyze(s);
  const std::vector<int> onsets =
      DetectOnsets(a.Feature(kFlux), a.num_frames, c, rate);
  Expect(onsets.size() == 3,
         "three notes give three onsets (got " +
             std::to_string(onsets.size()) + ")");
}

void TestCompareIdentical() {
  StftConfig c{256, 64};
  CpuAnalyzer cpu(c);
  const SignalAnalysis a = cpu.Analyze(MakeTone(500.0, 0.5, 8000, 0.3));
  SignalAnalysis b = a;
  Expect(CompareAnalyses(a, b, c).passed, "identical results pass");
  b.features[kRms * b.num_frames] *= 1.01f;
  Expect(!CompareAnalyses(a, b, c).passed, "1% RMS error fails");
}

void TestOptions() {
  const char* argv[] = {"prog", "--streams=8", "--frame_length", "2048",
                        "--no_cpu_check", "--input_dir=x"};
  Options o;
  std::string error;
  Expect(ParseOptions(6, const_cast<char**>(argv), &o, &error),
         "valid flags parse");
  Expect(o.streams == 8 && o.frame_length == 2048 && !o.cpu_check &&
             o.labels_path == "x/labels.csv",
         "flag values are stored");
  const char* bad[] = {"prog", "--frame_length=1000"};
  Options o2;
  Expect(!ParseOptions(2, const_cast<char**>(bad), &o2, &error),
         "non power-of-two frame is rejected");
}

}  // namespace
}  // namespace audio

int main() {
  audio::TestFrameCount();
  audio::TestFftMatchesDft();
  audio::TestWavRoundTrip();
  audio::TestDominantFrequency();
  audio::TestOnsets();
  audio::TestCompareIdentical();
  audio::TestOptions();
  if (audio::g_failures == 0) {
    std::printf("All host tests passed.\n");
    return 0;
  }
  std::printf("%d host test(s) failed.\n", audio::g_failures);
  return 1;
}
