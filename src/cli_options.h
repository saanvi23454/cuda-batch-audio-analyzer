// Command-line options of the audio_analyzer program.

#ifndef SRC_CLI_OPTIONS_H_
#define SRC_CLI_OPTIONS_H_

#include <string>

namespace audio {

struct Options {
  std::string input_dir = "data/input";
  std::string output_dir = "results/output";
  std::string labels_path;  // Defaults to <input_dir>/labels.csv.
  int frame_length = 1024;
  int hop = 256;
  int streams = 4;
  int max_files = 0;   // 0 means all files.
  int save_count = 8;  // Signals whose spectrogram and frame CSV are saved.
  bool cpu_check = true;
  bool show_help = false;
};

// Parses "--name=value" or "--name value" arguments. Returns false and sets
// |error| for an unknown flag or an invalid value.
bool ParseOptions(int argc, char** argv, Options* options, std::string* error);

std::string UsageText(const char* program);

}  // namespace audio

#endif  // SRC_CLI_OPTIONS_H_
