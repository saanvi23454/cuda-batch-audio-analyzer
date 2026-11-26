#!/usr/bin/env python3
"""Packs the proof of execution into proof_of_execution.tar.gz.

The archive holds the run log, nvidia-smi output, CSV results, all figures
(before/after images) and the WAV files of the signals whose spectrograms
were saved, so a reviewer can listen to the "before" audio.

Usage: python3 scripts/package_proof.py --results results --input data/input
"""

import argparse
import glob
import os
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--input", default="data/input")
    parser.add_argument("--out", default="proof_of_execution.tar.gz")
    args = parser.parse_args()

    files = []
    for pattern in ["*.txt", "output/*.csv", "output/frames/*.csv",
                    "scaling/*/batch_timing.csv", "figures/*.png"]:
        files += sorted(glob.glob(os.path.join(args.results, pattern)))
    saved = glob.glob(os.path.join(args.results, "output", "spectrograms",
                                   "*_spectrogram.pgm"))
    for pgm in sorted(saved):
        stem = os.path.basename(pgm)[:-len("_spectrogram.pgm")]
        wav = os.path.join(args.input, stem + ".wav")
        if os.path.exists(wav):
            files.append(wav)

    with tarfile.open(args.out, "w:gz") as tar:
        for path in files:
            if os.path.abspath(path).startswith(os.path.abspath(args.input)):
                arcname = os.path.join("proof", "input_before",
                                       os.path.basename(path))
            else:
                arcname = os.path.join("proof",
                                       os.path.relpath(path, args.results))
            tar.add(path, arcname=arcname)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"Wrote {args.out} ({len(files)} files, {size_kb:.0f} KiB)")


if __name__ == "__main__":
    main()
