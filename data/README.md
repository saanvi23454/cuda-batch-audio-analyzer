# Data

`data/input/` holds the WAV files the analyzer reads. They are not committed
because they are large (about 42 MB for 400 files) and can be regenerated
exactly:

```bash
python3 scripts/generate_signals.py --out data/input --count 400 --seed 2024
```

This also writes `data/input/labels.csv` (`file,class,f0_hz,n_notes`), the
ground truth used to check the results. `run.sh` runs the generator
automatically when the folder has no `.wav` files.

To analyze your own recordings instead, put 16-bit PCM `.wav` files in a
folder and run `INPUT=/path/to/folder ./run.sh`.
