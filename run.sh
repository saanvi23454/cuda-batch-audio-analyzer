#!/usr/bin/env bash
# One command for the whole project:
#   dataset -> build -> unit tests -> GPU run (with CPU check) ->
#   stream-scaling runs -> figures -> proof_of_execution.tar.gz
#
# Environment overrides:
#   INPUT=dir     use your own folder of 16-bit PCM .wav files
#   COUNT=N       number of synthetic signals to generate (default 400)
#   STREAMS=N     CUDA streams for the main run (default 4)
# Extra arguments are passed to bin/audio_analyzer, e.g. ./run.sh --hop=512
set -euo pipefail
cd "$(dirname "$0")"

INPUT=${INPUT:-data/input}
COUNT=${COUNT:-400}
STREAMS=${STREAMS:-4}
RESULTS=results

echo "== 1. Dataset =="
if ls "$INPUT"/*.wav >/dev/null 2>&1; then
  echo "Using existing WAV files in $INPUT"
else
  python3 scripts/generate_signals.py --out "$INPUT" --count "$COUNT"
fi

echo "== 2. Build =="
make

mkdir -p "$RESULTS"
echo "== 3. Host unit tests =="
make test | tee "$RESULTS/host_tests.txt"

(nvidia-smi || echo "nvidia-smi not available") > "$RESULTS/nvidia_smi.txt" 2>&1

echo "== 4. GPU run =="
set +e
./bin/audio_analyzer --input_dir="$INPUT" --output_dir="$RESULTS/output" \
  --streams="$STREAMS" "$@" 2>&1 | tee "$RESULTS/run_log.txt"
status=${PIPESTATUS[0]}
set -e

echo "== 5. Stream scaling (GPU only) =="
: > "$RESULTS/stream_scaling_log.txt"
for s in 1 2 4 8; do
  ./bin/audio_analyzer --input_dir="$INPUT" \
    --output_dir="$RESULTS/scaling/streams_$s" --streams="$s" \
    --no_cpu_check --save_count=0 "$@" > "$RESULTS/scaling/log_$s.txt" 2>&1 \
    || echo "streams=$s run failed, see $RESULTS/scaling/log_$s.txt"
  echo "--- streams=$s ---" >> "$RESULTS/stream_scaling_log.txt"
  grep -E "^(GPU [0-9]|GPU batch|Signals)" "$RESULTS/scaling/log_$s.txt" \
    | tee -a "$RESULTS/stream_scaling_log.txt" || true
done

echo "== 6. Figures =="
python3 scripts/plot_results.py --results "$RESULTS" --input "$INPUT" \
  || echo "Plotting skipped (python3 -m pip install --user numpy matplotlib)"

echo "== 7. Proof of execution =="
python3 scripts/package_proof.py --results "$RESULTS" --input "$INPUT"

echo "Done. Main run exit code: $status (0 = all GPU results match the CPU)."
exit "$status"
