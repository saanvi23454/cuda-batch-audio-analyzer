// Optional ground-truth labels for a dataset (written by
// scripts/generate_signals.py as labels.csv).

#ifndef SRC_LABELS_H_
#define SRC_LABELS_H_

#include <map>
#include <string>

namespace audio {

struct Label {
  std::string kind;          // Signal class, e.g. "tone" or "melody".
  double f0_hz = -1.0;       // Expected dominant frequency; < 0 if unknown.
  int num_notes = -1;        // Expected number of onsets; < 0 if unknown.
};

// Reads "file,class,f0_hz,n_notes" rows (with a header line). Empty fields
// mean "unknown". Returns an empty map if the file does not exist.
std::map<std::string, Label> ReadLabels(const std::string& path);

}  // namespace audio

#endif  // SRC_LABELS_H_
