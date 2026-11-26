#include "cli_options.h"

#include <cstdlib>
#include <sstream>

namespace audio {
namespace {

bool ParseInt(const std::string& text, int min_value, int max_value,
              int* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  const long parsed = std::strtol(text.c_str(), &end, 10);
  if (*end != '\0' || parsed < min_value || parsed > max_value) return false;
  *value = static_cast<int>(parsed);
  return true;
}

bool IsPowerOfTwo(int v) { return v > 0 && (v & (v - 1)) == 0; }

}  // namespace

bool ParseOptions(int argc, char** argv, Options* options,
                  std::string* error) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.compare(0, 2, "--") != 0) {
      *error = "unexpected argument '" + arg + "'";
      return false;
    }
    arg = arg.substr(2);
    std::string name = arg;
    std::string value;
    bool has_value = false;
    const size_t eq = arg.find('=');
    if (eq != std::string::npos) {
      name = arg.substr(0, eq);
      value = arg.substr(eq + 1);
      has_value = true;
    }

    // Boolean flags.
    if (name == "help" || name == "h") {
      options->show_help = true;
      continue;
    }
    if (name == "no_cpu_check") {
      options->cpu_check = false;
      continue;
    }

    // All remaining flags need a value: "--name=value" or "--name value".
    if (!has_value) {
      if (i + 1 >= argc) {
        *error = "flag --" + name + " needs a value";
        return false;
      }
      value = argv[++i];
    }

    bool ok = true;
    if (name == "input_dir") {
      options->input_dir = value;
    } else if (name == "output_dir") {
      options->output_dir = value;
    } else if (name == "labels") {
      options->labels_path = value;
    } else if (name == "frame_length") {
      ok = ParseInt(value, 64, 16384, &options->frame_length) &&
           IsPowerOfTwo(options->frame_length);
    } else if (name == "hop") {
      ok = ParseInt(value, 1, 16384, &options->hop);
    } else if (name == "streams") {
      ok = ParseInt(value, 1, 32, &options->streams);
    } else if (name == "max_files") {
      ok = ParseInt(value, 0, 1000000, &options->max_files);
    } else if (name == "save_count") {
      ok = ParseInt(value, 0, 1000000, &options->save_count);
    } else {
      *error = "unknown flag --" + name;
      return false;
    }
    if (!ok) {
      *error = "invalid value '" + value + "' for --" + name;
      return false;
    }
  }
  if (options->hop > options->frame_length) {
    *error = "--hop must not be larger than --frame_length";
    return false;
  }
  if (options->labels_path.empty()) {
    options->labels_path = options->input_dir + "/labels.csv";
  }
  return true;
}

std::string UsageText(const char* program) {
  std::ostringstream out;
  out << "Usage: " << program << " [flags]\n\n"
      << "Batch STFT analysis of 16-bit PCM WAV files on the GPU "
         "(NPP + cuFFT + CUDA kernels).\n\n"
      << "  --input_dir=DIR      folder of .wav files (default data/input)\n"
      << "  --output_dir=DIR     where CSVs and spectrograms go "
         "(default results/output)\n"
      << "  --labels=FILE        ground-truth CSV (default "
         "<input_dir>/labels.csv)\n"
      << "  --frame_length=N     STFT frame size, power of two (default "
         "1024)\n"
      << "  --hop=N              samples between frames (default 256)\n"
      << "  --streams=N          concurrent CUDA streams, 1-32 (default 4)\n"
      << "  --max_files=N        only process the first N files (default "
         "all)\n"
      << "  --save_count=N       signals whose spectrogram and per-frame CSV "
         "are saved (default 8)\n"
      << "  --no_cpu_check       skip the single-threaded CPU reference\n"
      << "  --help               show this text\n\n"
      << "Exit code: 0 = all GPU results match the CPU reference, "
         "2 = mismatch, 1 = usage or I/O error.\n";
  return out.str();
}

}  // namespace audio
