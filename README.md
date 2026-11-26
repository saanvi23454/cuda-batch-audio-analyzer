# CUDA Batch Audio Analyzer: NPP + cuFFT Feature Extraction for Hundreds of Signals

Independent project for **CUDA at Scale for the Enterprise**.

The program takes a folder of WAV recordings (hundreds of short signals) and runs a short-time Fourier transform (STFT) analysis on every one of them on the GPU. For each signal it produces:

* a **spectrogram** (an 8-bit image of frequency content over time),
* four **per-frame features**: RMS loudness, zero-crossing rate, spectral centroid and spectral flux,
* the **dominant frequency** of the whole signal, and
* the **note onsets** (where new sounds start), found from the spectral flux.

The signal processing runs on the GPU with **NPP signal functions**, **batched cuFFT** and **custom CUDA kernels**. Several CUDA streams keep different signals in flight at once. A single-threaded CPU version of the same pipeline checks every GPU result and gives a speed-up baseline. The bundled dataset generator writes ground-truth labels, so the program can also check that its answers are actually right.

```
 WAV (int16) ──► NPP: Convert_16s32f ─► NPP: Mean, MinMax ─► kernel: remove DC, scale to ±1
                                                                       │
     ┌─────────────────────────────────────────────────────────────────┘
     ▼
 kernel: cut into Hann-windowed frames ─► cuFFT R2C (batched, 64 frames per call)
     │
     ▼
 kernel: power + dB ─┬─► kernel (1 block per frame): RMS, ZCR, centroid, flux ─► onsets (host)
                     ├─► kernel: mean spectrum ─► NPP: MaxIndx ─► dominant frequency
                     └─► NPP: Max ─► kernel: 8-bit spectrogram image
```

## How the GPU is used

| Step | GPU code | Notes |
|---|---|---|
| int16 → float | `nppsConvert_16s32f_Ctx` | NPP signal conversion |
| DC offset, peak | `nppsMean_32f_Ctx`, `nppsMinMax_32f_Ctx` | NPP reductions; the results stay on the GPU |
| Normalize | `NormalizeKernel` | reads the NPP results straight from device memory, so the host never waits |
| Framing + window | `FrameAndWindowKernel` | periodic Hann window, zero-padding for the last FFT chunk |
| FFT | `cufftPlanMany` + `cufftExecR2C` | one batched plan (64 frames) per stream, reused for signals of any length |
| Power / dB | `PowerSpectrumKernel` | `|X|²/N²` and `10·log10` |
| Frame features | `FrameFeaturesKernel` | one thread block per frame, warp-shuffle reductions of 5 sums |
| Mean spectrum | `MeanSpectrumKernel` | one thread per frequency bin, coalesced reads |
| Dominant bin | `nppsMaxIndx_32f_Ctx` | NPP arg-max over the mean spectrum (DC skipped) |
| Spectrogram image | `nppsMax_32f_Ctx` + `SpectrogramImageKernel` | 80 dB range below the loudest bin |

**Working at scale.** `GpuAnalyzer` owns *N* "lanes". Each lane has its own CUDA stream, NPP stream context, cuFFT plan, pinned host buffers, device buffers and timing events. Signals are handed to the lanes round-robin. Before a lane takes a new signal, the host collects the previous one while the other lanes keep the GPU busy. Everything is allocated once, up front, so the batch loop never calls `cudaMalloc`, which would make the streams wait for each other. `run.sh` repeats the batch with 1, 2, 4 and 8 streams to measure how much overlap helps.

## Dataset

`scripts/generate_signals.py` writes **400 WAV files** by default: 16 kHz mono, 1–6 s each, about 21 M samples and 22 minutes of audio. There are six signal classes, and the answer is known for three of them:

| Class | What it is | Label used for checking |
|---|---|---|
| `tone` | one sine wave plus noise | true frequency |
| `harmonic` | instrument-like note with 6 decaying harmonics | true fundamental |
| `chord` | three notes, one clearly louder | frequency of the loudest note |
| `melody` | 2–8 separate notes with short rests | number of notes |
| `chirp` | linear frequency sweep | none |
| `noise` | low-passed noise with 50 Hz hum | none |

Every file also gets random noise, gain and a DC offset, so the normalization stage has real work to do. The output is deterministic (`--seed`). You can also run the analyzer on **your own recordings**: any 16-bit PCM WAV files work, mono or multi-channel, at any sample rate (`INPUT=/path/to/wavs ./run.sh`). Without `labels.csv` the label checks are skipped.

## Repository layout

```
├── Makefile                 build (nvcc + g++), `make test`, `make run`
├── run.sh                   one command: data → build → tests → run → scaling → plots → proof
├── src/
│   ├── main.cc              CLI driver: load WAVs, GPU batch, CPU check, labels, CSV/PGM output
│   ├── gpu_analyzer.{h,cu}  lanes (streams), NPP calls, cuFFT plans, async copies
│   ├── kernels.{h,cu}       custom CUDA kernels and their launchers
│   ├── cpu_reference.{h,cc} single-threaded version of the full pipeline (radix-2 FFT)
│   ├── features.{h,cc}      onset picking, dominant-frequency refinement, GPU-vs-CPU comparison
│   ├── wav_io.{h,cc}        16-bit PCM WAV reader/writer (no external libraries)
│   ├── labels.{h,cc}        ground-truth CSV reader
│   ├── file_util.{h,cc}     directory listing, mkdir -p, PGM writer
│   ├── cli_options.{h,cc}   command-line parsing
│   ├── cuda_check.h         CUDA / cuFFT / NPP error checking
│   └── signal_types.h       shared structs (STFT layout, signal, analysis result)
├── tests/
│   ├── host_test.cc         unit tests: FFT vs DFT, WAV round trip, features, onsets, flags
│   └── reference_check.cc   CPU reference vs labels (used to tune onset detection)
├── scripts/
│   ├── generate_signals.py  labelled synthetic dataset
│   ├── plot_results.py      before/after figures, timing, stream scaling, f0 accuracy
│   └── package_proof.py     builds proof_of_execution.tar.gz
├── data/input/              WAV files (generated by run.sh, not committed)
└── results/                 proof of execution from the GPU run
```

## Requirements

| Need | Notes |
|---|---|
| NVIDIA GPU + driver | any CUDA GPU from Pascal (sm_60) on |
| CUDA Toolkit 11.x / 12.x / 13.x | `nvcc`, cuFFT and NPP (`libnppc`, `libnpps`) all ship with the toolkit |
| g++ with C++14 | for the host code |
| Python 3 + numpy | dataset generator; `matplotlib` for the figures |

```bash
python3 -m pip install --user numpy matplotlib   # only if they are missing
```

If CUDA is not in `/usr/local/cuda`, pass `CUDA_PATH=/path/to/cuda` to `make`. The Makefile builds for `sm_60` (with PTX, so newer GPUs JIT-compile it), or `sm_75` with CUDA 13. Use `ARCH=sm_86` (for example) to target one GPU.

## Build and run

Everything in one command (this is what produced `results/`):

```bash
./run.sh                    # 400 synthetic signals, 4 streams
COUNT=1000 ./run.sh         # more data
STREAMS=8 ./run.sh --hop=512
INPUT=~/my_wavs ./run.sh    # your own recordings
```

Step by step:

```bash
make                        # builds bin/audio_analyzer
make test                   # host unit tests, no GPU needed
python3 scripts/generate_signals.py --out data/input --count 400
./bin/audio_analyzer --input_dir=data/input --output_dir=results/output --streams=4
python3 scripts/plot_results.py --results results --input data/input
python3 scripts/package_proof.py --results results --input data/input
```

### Command-line flags

| Flag | Default | Meaning |
|---|---|---|
| `--input_dir=DIR` | `data/input` | folder of `.wav` files |
| `--output_dir=DIR` | `results/output` | where CSVs, frame data and spectrograms go |
| `--labels=FILE` | `<input_dir>/labels.csv` | ground truth (optional) |
| `--frame_length=N` | `1024` | STFT frame size (power of two, 64–16384) |
| `--hop=N` | `256` | samples between frames |
| `--streams=N` | `4` | concurrent CUDA streams (1–32) |
| `--max_files=N` | all | process only the first N files |
| `--save_count=N` | `8` | signals (spread over the batch) whose spectrogram image and per-frame CSV are saved |
| `--no_cpu_check` | | skip the CPU reference (GPU timing only) |
| `--help` | | print usage |

Exit code: `0` when every GPU result matches the CPU reference, `2` when any signal does not, and `1` for usage or I/O errors.

## Output

| File | Contents |
|---|---|
| `results/run_log.txt` | full console output: GPU, library versions, one line per signal, summary |
| `results/nvidia_smi.txt` | the GPU the code ran on |
| `results/host_tests.txt` | unit test output |
| `results/stream_scaling_log.txt` | batch time with 1, 2, 4 and 8 streams |
| `results/output/summary.csv` | per signal: GPU/CPU time, dominant frequency, mean features, onsets, label checks, PASS/FAIL |
| `results/output/batch_timing.csv` | batch wall time, throughput, speed-up, setup time |
| `results/output/frames/*_frames.csv` | per-frame RMS, ZCR, centroid, flux and onset flag |
| `results/output/spectrograms/*.pgm` | 8-bit GPU spectrograms |
| `results/figures/*_before_after.png` | **before** (input waveform) and **after** (GPU spectrogram with onsets, feature curves) |
| `results/figures/timing_per_signal.png`, `stream_scaling.png`, `f0_vs_label.png` | summary plots |
| `proof_of_execution.tar.gz` | all of the above plus the "before" WAV files, for the Coursera upload |

## How results are checked

1. **GPU vs CPU.** The CPU reference repeats every stage in double precision with its own radix-2 FFT. For each signal the program requires the same peak bin (or a tie within 1e-4), RMS, centroid and flux within 0.1 % of the largest value, the mean spectrum within 0.1 %, and the zero-crossing rate within 4/(N−1). That last allowance covers a sample that is within rounding of zero landing on the other side.
2. **Against the truth.** For tones, harmonic notes and chords, the dominant frequency must be within half an FFT bin (7.8 Hz) of the true frequency. For melodies, the number of detected onsets must equal the number of notes. I checked both with the CPU reference before the GPU run (`make reference_check`). On the default dataset the dominant frequency was right for 250/250 signals and the onset count for 97/97. On a second dataset with a different seed (200 files) the results were 128/128 and 44/44.

## Design decisions and lessons learned

* **Onset detection needed an adaptive threshold.** My first version marked a frame as an onset when its flux was above 15 % of the signal's maximum. Against the labels it got only 52 of 97 melodies right, almost always by finding too many onsets. Spectral flux sums positive changes over all 513 bins, so even quiet background noise adds a steady amount of flux between notes. Subtracting a moving median (±0.2 s) before thresholding removed that baseline and fixed every case. Without ground-truth labels I would never have noticed, since the GPU and CPU agreed perfectly on the wrong answer.
* **One cuFFT plan for every signal length.** A batched plan has a fixed batch size, and the signals have different lengths. Rather than one plan per length, each lane has a single plan for 64 frames and calls it once per 64-frame chunk. The framing kernel zero-fills the padding frames.
* **No host round trips inside the pipeline.** The NPP reductions (mean, min/max, max, arg-max) write their results to device memory, and the next kernel reads them from there. The host never copies a scalar back in the middle of a signal, so a whole signal is one uninterrupted sequence of work on its stream.
* **NPP's integer types differ between CUDA versions.** The signal functions take `int` lengths in CUDA 11 and `size_t` in newer releases, and `MaxIndx`'s index type changes too. `gpu_analyzer.cu` works out these types from the function signatures with small templates, so the same source builds against either version.
* **Pinned memory and a warm-up.** Asynchronous copies only overlap with kernels when the host buffer is pinned. A one-signal warm-up run happens before timing, so lazy module loading and PTX JIT are not counted in the batch time. They are reported separately as setup time.
* **Style.** The code follows the Google C++ Style Guide (`.clang-format` is `BasedOnStyle: Google`, 80 columns, `CamelCase` functions, `snake_case_` members, `kConstants`, no exceptions).

## Results

See `results/run_log.txt` and the figures in `results/figures/` after running `./run.sh` on a GPU.

## Attribution

All audio is synthetic and made by `scripts/generate_signals.py`. The project structure follows the course template: https://github.com/PascaleCourseraCourses/CUDAatScaleForTheEnterpriseCourseProjectTemplate
