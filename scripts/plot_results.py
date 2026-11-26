#!/usr/bin/env python3
"""Turns the analyzer's CSV and PGM output into figures.

  results/figures/<signal>_before_after.png  waveform (input) vs. GPU
                                             spectrogram, onsets, features
  results/figures/timing_per_signal.png      GPU vs CPU time per signal
  results/figures/stream_scaling.png         throughput vs number of streams
  results/figures/f0_vs_label.png            measured vs true frequency

Usage: python3 scripts/plot_results.py --results results --input data/input
"""

import argparse
import csv
import glob
import os
import wave

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_pgm(path):
    with open(path, "rb") as f:
        data = f.read()
    tokens, pos = [], 0
    while len(tokens) < 4:  # P5, width, height, maxval
        while data[pos:pos + 1].isspace():
            pos += 1
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    width, height = int(tokens[1]), int(tokens[2])
    pixels = np.frombuffer(data[pos + 1:pos + 1 + width * height], np.uint8)
    return pixels.reshape(height, width)


def read_wav(path):
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
        x = np.frombuffer(raw, "<i2").astype(np.float32)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    return x / 32768.0, rate


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_signal(stem, args, summary_row, out_dir):
    wav_path = os.path.join(args.input, stem + ".wav")
    pgm_path = os.path.join(args.output, "spectrograms",
                            stem + "_spectrogram.pgm")
    csv_path = os.path.join(args.output, "frames", stem + "_frames.csv")
    if not (os.path.exists(wav_path) and os.path.exists(pgm_path)
            and os.path.exists(csv_path)):
        return None
    x, rate = read_wav(wav_path)
    image = read_pgm(pgm_path)
    frames = read_csv(csv_path)
    times = np.array([float(r["time_s"]) for r in frames])
    rms = np.array([float(r["rms"]) for r in frames])
    centroid = np.array([float(r["centroid_hz"]) for r in frames])
    onsets = [float(r["time_s"]) for r in frames if r["onset"] == "1"]
    duration = len(x) / rate

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2, 1]})
    axes[0].plot(np.arange(len(x)) / rate, x, lw=0.4, color="#335")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title(f"BEFORE: input waveform  {stem}.wav")

    axes[1].imshow(image, aspect="auto", cmap="magma",
                   extent=[0, duration, 0, rate / 2])
    for t in onsets:
        axes[1].axvline(t, color="cyan", lw=0.8, ls="--")
    label = summary_row.get("label_class", "") if summary_row else ""
    f0 = float(summary_row["dominant_hz"]) if summary_row else 0.0
    axes[1].set_ylabel("frequency (Hz)")
    axes[1].set_title(f"AFTER: GPU spectrogram (80 dB range), "
                      f"dominant {f0:.1f} Hz, {len(onsets)} onsets (dashed)"
                      + (f", class '{label}'" if label else ""))

    axes[2].plot(times, rms, color="#c33", label="RMS")
    axes[2].set_ylabel("RMS")
    twin = axes[2].twinx()
    twin.plot(times, centroid, color="#36c", label="centroid")
    twin.set_ylabel("centroid (Hz)")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("GPU per-frame features")
    fig.tight_layout()
    path = os.path.join(out_dir, stem + "_before_after.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_timing(summary, out_dir):
    duration = np.array([float(r["duration_s"]) for r in summary])
    gpu = np.array([float(r["gpu_ms"]) for r in summary])
    cpu = np.array([float(r["cpu_ms"]) for r in summary])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(duration, gpu, s=8, label="GPU latency (copies + kernels)")
    if np.any(cpu > 0):
        ax.scatter(duration, cpu, s=8, label="CPU reference (1 thread)")
    ax.set_yscale("log")
    ax.set_xlabel("signal length (s)")
    ax.set_ylabel("time per signal (ms)")
    ax.set_title("Per-signal processing time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "timing_per_signal.png"), dpi=110)
    plt.close(fig)


def plot_scaling(results_dir, out_dir):
    rows = []
    for path in glob.glob(os.path.join(results_dir, "scaling", "*",
                                       "batch_timing.csv")):
        rows.extend(read_csv(path))
    if not rows:
        return
    rows.sort(key=lambda r: int(r["streams"]))
    streams = [r["streams"] for r in rows]
    throughput = [float(r["gpu_msamples_per_s"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(streams, throughput, color="#4a7")
    for i, v in enumerate(throughput):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    ax.set_xlabel("CUDA streams")
    ax.set_ylabel("M samples / s (whole batch)")
    ax.set_title("Throughput vs. number of streams")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stream_scaling.png"), dpi=110)
    plt.close(fig)


def plot_f0(summary, out_dir):
    rows = [r for r in summary if r.get("label_f0_hz")]
    if not rows:
        return
    true = np.array([float(r["label_f0_hz"]) for r in rows])
    found = np.array([float(r["dominant_hz"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(true, found, s=8)
    lim = [0, max(true.max(), found.max()) * 1.05]
    ax.plot(lim, lim, color="gray", lw=0.8)
    ax.set_xlabel("true f0 (Hz)")
    ax.set_ylabel("GPU dominant frequency (Hz)")
    ok = sum(r["f0_ok"] == "1" for r in rows)
    ax.set_title(f"Dominant frequency: {ok}/{len(rows)} within half a bin")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f0_vs_label.png"), dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default=None,
                        help="analyzer output dir (default <results>/output)")
    parser.add_argument("--input", default="data/input")
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(args.results, "output")

    out_dir = os.path.join(args.results, "figures")
    os.makedirs(out_dir, exist_ok=True)
    summary = read_csv(os.path.join(args.output, "summary.csv"))
    by_stem = {os.path.splitext(r["file"])[0]: r for r in summary}

    written = 0
    for pgm in sorted(glob.glob(os.path.join(args.output, "spectrograms",
                                             "*_spectrogram.pgm"))):
        stem = os.path.basename(pgm)[:-len("_spectrogram.pgm")]
        if plot_signal(stem, args, by_stem.get(stem), out_dir):
            written += 1
    plot_timing(summary, out_dir)
    plot_scaling(args.results, out_dir)
    plot_f0(summary, out_dir)
    print(f"Wrote {written} before/after figures and summary plots to "
          f"{out_dir}")


if __name__ == "__main__":
    main()
