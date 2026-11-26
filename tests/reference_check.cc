// Runs the CPU reference on a labelled dataset and reports how often it
// finds the right dominant frequency and number of notes. This checks the
// reference and the onset settings themselves; it needs no GPU.
//
//   bin/reference_check data/input [max_files]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "cpu_reference.h"
#include "feature_utils.h"
#include "file_util.h"
#include "labels.h"
#include "wav_io.h"

int main(int argc, char** argv) {
  using namespace audio;  // NOLINT(build/namespaces): small test tool.
  const std::string dir = argc > 1 ? argv[1] : "data/input";
  const int limit = argc > 2 ? std::atoi(argv[2]) : 0;
  const std::map<std::string, Label> labels = ReadLabels(dir + "/labels.csv");
  std::vector<std::string> names = ListFiles(dir, ".wav");
  if (limit > 0 && names.size() > static_cast<size_t>(limit)) {
    names.resize(limit);
  }

  const StftConfig config{1024, 256};
  CpuAnalyzer cpu(config);
  int f0_total = 0;
  int f0_ok = 0;
  int notes_total = 0;
  int notes_ok = 0;
  for (const std::string& name : names) {
    Signal signal;
    std::string error;
    if (!ReadWav(JoinPath(dir, name), &signal, &error)) continue;
    const auto label = labels.find(name);
    if (label == labels.end()) continue;
    const SignalAnalysis a = cpu.Analyze(signal);
    if (label->second.f0_hz > 0) {
      const double hz = DominantFrequencyHz(a.mean_spectrum, a.peak_bin,
                                            config, signal.sample_rate);
      ++f0_total;
      if (std::fabs(hz - label->second.f0_hz) <=
          0.5 * config.BinHz(signal.sample_rate)) {
        ++f0_ok;
      } else {
        std::printf("f0 miss  %s: %.1f Hz, expected %.1f\n", name.c_str(), hz,
                    label->second.f0_hz);
      }
    }
    if (label->second.num_notes > 0) {
      const std::vector<int> onsets = DetectOnsets(
          a.Feature(kFlux), a.num_frames, config, signal.sample_rate);
      ++notes_total;
      if (static_cast<int>(onsets.size()) == label->second.num_notes) {
        ++notes_ok;
      } else {
        std::printf("notes miss %s: %zu onsets, expected %d\n", name.c_str(),
                    onsets.size(), label->second.num_notes);
      }
    }
  }
  std::printf("Dominant frequency: %d/%d, onset count: %d/%d\n", f0_ok,
              f0_total, notes_ok, notes_total);
  return 0;
}
