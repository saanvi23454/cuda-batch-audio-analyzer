#!/usr/bin/env python3
"""Generates a labelled dataset of synthetic 16-bit mono WAV files.

Every file belongs to one of six classes. The ones with a known answer let
the program check its own output:

  tone      one sine wave plus noise                 -> f0 known
  harmonic  instrument-like note (decaying harmonics) -> f0 known
  chord     three notes, one clearly loudest          -> f0 known
  melody    2-8 separate notes with short rests       -> number of notes known
  chirp     linear frequency sweep                    -> no label
  noise     band-limited noise with a quiet hum       -> no label

labels.csv columns: file,class,f0_hz,n_notes (empty = unknown).
The output is deterministic for a given --seed.

Usage: python3 scripts/generate_signals.py --out data/input --count 400
"""

import argparse
import csv
import os
import wave

import numpy as np

CLASSES = ["tone", "harmonic", "chord", "melody", "chirp", "noise"]
WEIGHTS = [0.20, 0.25, 0.15, 0.25, 0.08, 0.07]
# Pentatonic scale notes between ~200 Hz and ~1 kHz for the melodies.
MELODY_NOTES = [196.0, 220.0, 261.6, 293.7, 329.6, 392.0, 440.0, 523.3,
                587.3, 659.3, 784.0, 880.0, 987.8]


def envelope(n, rate, attack=0.01, decay=0.35):
    """Fast attack followed by an exponential decay."""
    t = np.arange(n) / rate
    env = np.exp(-t / decay)
    a = max(1, int(attack * rate))
    env[:a] *= np.linspace(0.0, 1.0, a)
    return env


def make_tone(rng, n, rate):
    f0 = rng.uniform(100.0, 3000.0)
    t = np.arange(n) / rate
    x = np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    return x, f0, None


def make_harmonic(rng, n, rate):
    f0 = rng.uniform(110.0, 880.0)
    t = np.arange(n) / rate
    x = np.zeros(n)
    for k in range(1, 7):
        if k * f0 < rate / 2 - 200:
            x += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6))
    # A slow decay keeps the note audible over the whole file.
    x *= envelope(n, rate, attack=0.02, decay=n / rate)
    return x, f0, None


def make_chord(rng, n, rate):
    t = np.arange(n) / rate
    while True:
        freqs = np.sort(rng.uniform(150.0, 2500.0, size=3))
        if np.min(np.diff(freqs)) > 80.0:  # Keep the peaks well apart.
            break
    main = rng.integers(0, 3)
    x = np.zeros(n)
    amplitudes = [0.45, 0.35]
    for i, f in enumerate(freqs):
        a = 1.0 if i == main else amplitudes.pop()
        x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6))
    return x, freqs[main], None


def make_melody(rng, rate):
    notes = int(rng.integers(2, 9))
    pieces = [np.zeros(int(rng.uniform(0.10, 0.30) * rate))]  # Lead-in rest.
    for _ in range(notes):
        f0 = MELODY_NOTES[rng.integers(0, len(MELODY_NOTES))]
        length = int(rng.uniform(0.20, 0.45) * rate)
        t = np.arange(length) / rate
        note = (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(4 * np.pi * f0 * t)
                + 0.25 * np.sin(6 * np.pi * f0 * t))
        note *= rng.uniform(0.5, 1.0) * envelope(length, rate, decay=0.12)
        pieces.append(note)
        pieces.append(np.zeros(int(rng.uniform(0.06, 0.15) * rate)))  # Rest.
    return np.concatenate(pieces), None, notes


def make_chirp(rng, n, rate):
    f_start, f_end = rng.uniform(100.0, 1000.0), rng.uniform(1500.0, 6000.0)
    if rng.random() < 0.5:
        f_start, f_end = f_end, f_start
    t = np.arange(n) / rate
    duration = n / rate
    phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) / duration * t * t)
    return np.sin(phase), None, None


def make_noise(rng, n, rate):
    white = rng.standard_normal(n + 64)
    kernel = np.hanning(33)
    kernel /= kernel.sum()
    x = np.convolve(white, kernel, mode="same")[:n]  # Low-pass the noise.
    t = np.arange(n) / rate
    x += 0.05 * np.sin(2 * np.pi * 50.0 * t)  # Mains hum.
    return x, None, None


def to_int16(x, rng, noise_level):
    x = x / (np.max(np.abs(x)) + 1e-12)
    x = x * rng.uniform(0.3, 0.9)
    x = x + noise_level * rng.standard_normal(len(x))
    x = x + rng.uniform(-0.02, 0.02)  # Small DC offset, removed by stage 1.
    return np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)


def write_wav(path, samples, rate):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype("<i2").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/input")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--min_seconds", type=float, default=1.0)
    parser.add_argument("--max_seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    total_samples = 0
    for i in range(args.count):
        kind = CLASSES[rng.choice(len(CLASSES), p=WEIGHTS)]
        n = int(rng.uniform(args.min_seconds, args.max_seconds) * args.rate)
        if kind == "tone":
            x, f0, notes = make_tone(rng, n, args.rate)
        elif kind == "harmonic":
            x, f0, notes = make_harmonic(rng, n, args.rate)
        elif kind == "chord":
            x, f0, notes = make_chord(rng, n, args.rate)
        elif kind == "melody":
            x, f0, notes = make_melody(rng, args.rate)
        elif kind == "chirp":
            x, f0, notes = make_chirp(rng, n, args.rate)
        else:
            x, f0, notes = make_noise(rng, n, args.rate)

        samples = to_int16(x, rng, noise_level=rng.uniform(0.002, 0.02))
        name = f"sig_{i:04d}_{kind}.wav"
        write_wav(os.path.join(args.out, name), samples, args.rate)
        total_samples += len(samples)
        rows.append([name, kind, "" if f0 is None else f"{f0:.3f}",
                     "" if notes is None else str(notes)])

    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "class", "f0_hz", "n_notes"])
        writer.writerows(rows)
    print(f"Wrote {args.count} WAV files ({total_samples / 1e6:.2f} M samples,"
          f" {total_samples / args.rate:.1f} s of audio) and labels.csv to"
          f" {args.out}")


if __name__ == "__main__":
    main()
