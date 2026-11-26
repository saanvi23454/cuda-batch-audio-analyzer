// audio_analyzer: batch STFT feature extraction for a folder of WAV files on
// the GPU, checked against a single-threaded CPU reference.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "cli_options.h"
#include "cpu_reference.h"
#include "feature_utils.h"
#include "file_util.h"
#include "gpu_analyzer.h"
#include "labels.h"
#include "signal_types.h"
#include "wav_io.h"

namespace audio {
namespace {

using Clock = std::chrono::steady_clock;

double MillisecondsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

// Results of one signal after host post-processing.
struct SignalReport {
  double dominant_hz = 0.0;
  std::vector<int> onsets;
  double cpu_ms = 0.0;
  bool checked = false;  // CPU comparison done.
  Comparison comparison;
  bool f0_checked = false;
  bool f0_ok = false;
  bool notes_checked = false;
  bool notes_ok = false;
};

// Picks |count| signals spread evenly over the batch for saved outputs.
std::vector<bool> ChooseSaved(size_t total, int count) {
  std::vector<bool> chosen(total, false);
  if (total == 0 || count <= 0) return chosen;
  const size_t n = std::min(total, static_cast<size_t>(count));
  for (size_t i = 0; i < n; ++i) chosen[i * total / n] = true;
  return chosen;
}

bool WriteFrameCsv(const std::string& path, const SignalAnalysis& analysis,
                   const std::vector<int>& onsets, const StftConfig& config,
                   int sample_rate) {
  std::ofstream out(path);
  if (!out) return false;
  std::vector<bool> is_onset(analysis.num_frames, false);
  for (int t : onsets) is_onset[t] = true;
  out << "frame,time_s,rms,zcr,centroid_hz,flux,onset\n";
  for (int t = 0; t < analysis.num_frames; ++t) {
    const double time =
        (t * config.hop + config.frame_length / 2.0) / sample_rate;
    out << t << "," << time << "," << analysis.Feature(kRms)[t] << ","
        << analysis.Feature(kZeroCrossingRate)[t] << ","
        << analysis.Feature(kCentroidHz)[t] << ","
        << analysis.Feature(kFlux)[t] << "," << (is_onset[t] ? 1 : 0)
        << "\n";
  }
  return static_cast<bool>(out);
}

int Run(const Options& options) {
  const StftConfig config{options.frame_length, options.hop};

  // ---- Load the dataset ----------------------------------------------
  std::vector<std::string> names = ListFiles(options.input_dir, ".wav");
  if (options.max_files > 0 &&
      names.size() > static_cast<size_t>(options.max_files)) {
    names.resize(options.max_files);
  }
  if (names.empty()) {
    std::fprintf(stderr, "No .wav files found in %s\n",
                 options.input_dir.c_str());
    return 1;
  }

  const Clock::time_point load_start = Clock::now();
  std::vector<Signal> signals;
  signals.reserve(names.size());
  for (const std::string& name : names) {
    Signal signal;
    std::string error;
    if (ReadWav(JoinPath(options.input_dir, name), &signal, &error)) {
      signals.push_back(std::move(signal));
    } else {
      std::fprintf(stderr, "Skipping %s: %s\n", name.c_str(), error.c_str());
    }
  }
  if (signals.empty()) {
    std::fprintf(stderr, "No readable 16-bit PCM WAV files.\n");
    return 1;
  }
  const double load_ms = MillisecondsSince(load_start);

  size_t total_samples = 0;
  size_t max_samples = 0;
  size_t min_samples = signals[0].samples.size();
  double total_seconds = 0.0;
  for (const Signal& s : signals) {
    total_samples += s.samples.size();
    max_samples = std::max(max_samples, s.samples.size());
    min_samples = std::min(min_samples, s.samples.size());
    total_seconds += static_cast<double>(s.samples.size()) / s.sample_rate;
  }
  const std::map<std::string, Label> labels =
      ReadLabels(options.labels_path);

  std::printf("=== CUDA batch audio analyzer ===\n");
  std::printf("Input: %s, %zu signals, %.2f M samples, %.1f s of audio "
              "(shortest %zu, longest %zu samples), loaded in %.1f ms\n",
              options.input_dir.c_str(), signals.size(), total_samples / 1e6,
              total_seconds, min_samples, max_samples, load_ms);
  std::printf("STFT: frame %d, hop %d, %d bins. Labels: %s (%zu rows)\n",
              config.frame_length, config.hop, config.NumBins(),
              labels.empty() ? "none" : options.labels_path.c_str(),
              labels.size());

  // ---- GPU -------------------------------------------------------------
  const Clock::time_point setup_start = Clock::now();
  GpuAnalyzer gpu(config, max_samples, options.streams);
  const double setup_ms = MillisecondsSince(setup_start);
  std::printf("%s\n", gpu.DeviceDescription().c_str());
  std::printf("GPU setup (context, buffers, cuFFT plans): %.1f ms\n",
              setup_ms);

  // Warm-up on one signal so lazy module loading is not timed.
  {
    std::vector<SignalAnalysis> warm_up;
    gpu.AnalyzeBatch(std::vector<Signal>(1, signals[0]),
                     std::vector<bool>(1, false), &warm_up);
  }

  const std::vector<bool> saved =
      ChooseSaved(signals.size(), options.save_count);
  std::vector<SignalAnalysis> gpu_results;
  const Clock::time_point gpu_start = Clock::now();
  gpu.AnalyzeBatch(signals, saved, &gpu_results);
  const double gpu_wall_ms = MillisecondsSince(gpu_start);

  // ---- CPU reference and per-signal report ------------------------------
  CpuAnalyzer cpu(config);
  std::vector<SignalReport> reports(signals.size());
  double cpu_total_ms = 0.0;
  int passed = 0;
  int f0_checked = 0;
  int f0_ok = 0;
  int notes_checked = 0;
  int notes_ok = 0;

  std::printf("\n%-28s %7s %6s %8s %9s %7s %6s %9s %s\n", "signal", "sec",
              "frames", "gpu_ms", "f0_hz", "rms", "onsets", "label",
              "check");
  for (size_t i = 0; i < signals.size(); ++i) {
    const Signal& signal = signals[i];
    const SignalAnalysis& g = gpu_results[i];
    SignalReport& report = reports[i];
    report.dominant_hz =
        DominantFrequencyHz(g.mean_spectrum, g.peak_bin, config,
                            signal.sample_rate);
    report.onsets = DetectOnsets(g.Feature(kFlux), g.num_frames, config,
                                 signal.sample_rate);

    if (options.cpu_check) {
      const Clock::time_point cpu_start = Clock::now();
      const SignalAnalysis c = cpu.Analyze(signal);
      report.cpu_ms = MillisecondsSince(cpu_start);
      cpu_total_ms += report.cpu_ms;
      report.checked = true;
      report.comparison = CompareAnalyses(g, c, config);
      if (report.comparison.passed) ++passed;
    }

    std::string label_text = "-";
    const auto label = labels.find(signal.name);
    if (label != labels.end()) {
      label_text = label->second.kind;
      if (label->second.f0_hz > 0.0) {
        report.f0_checked = true;
        // Within half a frequency bin of the true frequency.
        report.f0_ok = std::fabs(report.dominant_hz - label->second.f0_hz) <=
                       0.5 * config.BinHz(signal.sample_rate);
        ++f0_checked;
        if (report.f0_ok) ++f0_ok;
      }
      if (label->second.num_notes > 0) {
        report.notes_checked = true;
        report.notes_ok = static_cast<int>(report.onsets.size()) ==
                          label->second.num_notes;
        ++notes_checked;
        if (report.notes_ok) ++notes_ok;
      }
    }

    const char* check = !report.checked ? "gpu-only"
                        : report.comparison.passed ? "PASS"
                                                   : "FAIL";
    std::printf("%-28s %7.2f %6d %8.3f %9.1f %7.4f %6zu %9s %s\n",
                signal.name.c_str(),
                static_cast<double>(signal.samples.size()) /
                    signal.sample_rate,
                g.num_frames, g.gpu_ms, report.dominant_hz,
                MeanOf(g.Feature(kRms), g.num_frames), report.onsets.size(),
                label_text.c_str(), check);
    if (report.checked && !report.comparison.passed) {
      const Comparison& cmp = report.comparison;
      std::printf("    mismatch: same_peak=%d rms=%.2e zcr=%.2e "
                  "centroid=%.2e flux=%.2e spectrum=%.2e\n",
                  cmp.same_peak ? 1 : 0, cmp.feature_error[kRms],
                  cmp.feature_error[kZeroCrossingRate],
                  cmp.feature_error[kCentroidHz], cmp.feature_error[kFlux],
                  cmp.spectrum_error);
    }
  }

  // ---- Files ---------------------------------------------------------------
  const std::string frames_dir = options.output_dir + "/frames";
  const std::string images_dir = options.output_dir + "/spectrograms";
  if (!MakeDirectories(frames_dir) || !MakeDirectories(images_dir)) {
    std::fprintf(stderr, "Cannot create %s\n", options.output_dir.c_str());
    return 1;
  }

  std::ofstream summary(options.output_dir + "/summary.csv");
  summary << "file,sample_rate,samples,duration_s,frames,gpu_ms,cpu_ms,"
             "dominant_hz,peak_bin,mean_rms,mean_zcr,mean_centroid_hz,"
             "onsets,label_class,label_f0_hz,f0_ok,label_notes,onsets_ok,"
             "gpu_vs_cpu\n";
  int images_written = 0;
  for (size_t i = 0; i < signals.size(); ++i) {
    const Signal& signal = signals[i];
    const SignalAnalysis& g = gpu_results[i];
    const SignalReport& r = reports[i];
    const auto label = labels.find(signal.name);
    const bool has_label = label != labels.end();
    summary << signal.name << "," << signal.sample_rate << ","
            << signal.samples.size() << ","
            << static_cast<double>(signal.samples.size()) /
                   signal.sample_rate
            << "," << g.num_frames << "," << g.gpu_ms << "," << r.cpu_ms
            << "," << r.dominant_hz << "," << g.peak_bin << ","
            << MeanOf(g.Feature(kRms), g.num_frames) << ","
            << MeanOf(g.Feature(kZeroCrossingRate), g.num_frames) << ","
            << MeanOf(g.Feature(kCentroidHz), g.num_frames) << ","
            << r.onsets.size() << ","
            << (has_label ? label->second.kind : "") << ","
            << (r.f0_checked ? std::to_string(label->second.f0_hz) : "")
            << "," << (r.f0_checked ? (r.f0_ok ? "1" : "0") : "") << ","
            << (r.notes_checked ? std::to_string(label->second.num_notes)
                                : "")
            << "," << (r.notes_checked ? (r.notes_ok ? "1" : "0") : "")
            << ","
            << (!r.checked ? "skipped"
                           : (r.comparison.passed ? "PASS" : "FAIL"))
            << "\n";

    if (saved[i]) {
      const std::string stem = StripExtension(signal.name);
      WriteFrameCsv(frames_dir + "/" + stem + "_frames.csv", g, r.onsets,
                    config, signal.sample_rate);
      if (WritePgm(images_dir + "/" + stem + "_spectrogram.pgm",
                   g.num_frames, config.NumBins(), g.spectrogram)) {
        ++images_written;
      }
    }
  }

  const double gpu_msps = total_samples / (gpu_wall_ms * 1e3);
  std::ofstream batch(options.output_dir + "/batch_timing.csv");
  batch << "streams,signals,total_samples,gpu_wall_ms,gpu_msamples_per_s,"
           "cpu_ms,speedup,setup_ms\n"
        << options.streams << "," << signals.size() << "," << total_samples
        << "," << gpu_wall_ms << "," << gpu_msps << "," << cpu_total_ms
        << ","
        << (options.cpu_check && gpu_wall_ms > 0 ? cpu_total_ms / gpu_wall_ms
                                                 : 0.0)
        << "," << setup_ms << "\n";

  // ---- Summary -----------------------------------------------------------
  std::printf("\n=== Summary ===\n");
  std::printf("Signals: %zu (%.2f M samples, %.1f s of audio)\n",
              signals.size(), total_samples / 1e6, total_seconds);
  std::printf("GPU batch wall time (incl. all copies): %.2f ms -> "
              "%.1f M samples/s, %.0fx real time\n",
              gpu_wall_ms, gpu_msps, total_seconds * 1e3 / gpu_wall_ms);
  if (options.cpu_check) {
    std::printf("CPU reference (single thread): %.2f ms -> %.1f M "
                "samples/s\n",
                cpu_total_ms, total_samples / (cpu_total_ms * 1e3));
    std::printf("Speed-up (CPU / GPU batch): %.1fx\n",
                cpu_total_ms / gpu_wall_ms);
    std::printf("GPU vs CPU verification: %d/%zu PASS\n", passed,
                signals.size());
  }
  if (f0_checked > 0) {
    std::printf("Dominant frequency vs labels: %d/%d within half a bin\n",
                f0_ok, f0_checked);
  }
  if (notes_checked > 0) {
    std::printf("Onset count vs labels (melodies): %d/%d exact\n", notes_ok,
                notes_checked);
  }
  std::printf("Wrote %s/summary.csv, batch_timing.csv, %d spectrograms and "
              "frame CSVs\n",
              options.output_dir.c_str(), images_written);

  if (options.cpu_check && passed != static_cast<int>(signals.size())) {
    return 2;
  }
  return 0;
}

}  // namespace
}  // namespace audio

int main(int argc, char** argv) {
  audio::Options options;
  std::string error;
  if (!audio::ParseOptions(argc, argv, &options, &error)) {
    std::fprintf(stderr, "Error: %s\n\n%s", error.c_str(),
                 audio::UsageText(argv[0]).c_str());
    return 1;
  }
  if (options.show_help) {
    std::printf("%s", audio::UsageText(argv[0]).c_str());
    return 0;
  }
  return audio::Run(options);
}
