# wristlift

Automatic set detection and rep counting for strength training, using only the motion sensors of a
wrist-worn device (3-axis accelerometer, optionally a 3-axis gyroscope). No buttons to press: the
software notices when a set starts, counts the reps, notices when it ends, and starts the rest timer.

There are two implementations of the same algorithm:

| Folder / file | Language | Purpose |
|---|---|---|
| `wristlift.py` | Python | Reference implementation, offline analysis, training of optional models |
| `evaluate.py` | Python | Accuracy measurement and threshold tuning on labeled recordings |
| `zepp-app/` | JavaScript | Port for Zepp OS watches (target: Amazfit Active Edge) |

## How it works

1. **Data.** Accelerometer (and gyro if available) sampled at 25-50 Hz. Accelerometer values are
   auto-detected as g or m/s². Irregular sensor callbacks are resampled to a fixed rate.
2. **Set detection.** A 4 s window slides in 1 s steps. For each window the code computes motion energy,
   dominant frequency (0.2-1.3 Hz), spectral concentration and autocorrelation strength, and classifies it
   as exercising or idle. A set starts after 3 consecutive active windows (start time is back-dated) and
   ends after 7 s of idle. The end of a set is where the rest timer begins.
3. **Rep counting.**
   - The signal is low-pass filtered (6 Hz) and reduced to one dimension with PCA (accelerometer, or gyro
     if it is more periodic).
   - The rep period comes from the first strong autocorrelation peak.
   - Peaks are found with period-adapted spacing and prominence thresholds. Both polarities are tried,
     and the one whose count best matches (motion duration / period) is used, which avoids off-by-one
     errors at the start and end of a set.
   - Reps are only reported once at least 4 consistent reps exist; the count then jumps to the full
     number (the whole set is re-counted, so earlier reps are back-filled).
4. **Exercise recognition (optional).** `ExerciseClassifier` is a random forest over set-level features
   (mean wrist orientation, motion axis, range, variance). `build_cnn()` returns a Keras 1D-CNN for
   window-level recognition. **No trained model is included**; you supply labeled data.

## Quick start (Python)

Requirements: Python 3.9+, `numpy`, `scipy`, `scikit-learn` (TensorFlow only for the optional CNN).

```bash
python wristlift.py --demo                 # synthetic example
python wristlift.py recording.csv --fs 50  # columns: t,ax,ay,az[,gx,gy,gz]
```

As a library:

```python
from wristlift import WorkoutDetector

det = WorkoutDetector()
events = det.push(acc_chunk, gyro_chunk)   # arrays of shape (n, 3); call as samples arrive
events += det.flush()                      # at end of workout
```

Events:

| Event | Fields |
|---|---|
| `set_start` | `t` |
| `reps` | `t`, `count` (live count, only after 4 confirmed reps) |
| `set_end` | `start`, `end`, `reps`, `exercise`, `rest_starts` |
| `set_discarded` | `t` (motion that was not a real set) |

All tunable thresholds live in the `Config` dataclass (`min_energy`, `min_conc`, `min_ac`, `prom_frac`,
`end_idle_s`, ...).

## Measuring accuracy and tuning

`evaluate.py` compares detections with your ground truth.

Data layout, one folder:

```
session01.csv    t,ax,ay,az[,gx,gy,gz]
session01.json   {"sets": [{"start": 12.4, "end": 41.0, "reps": 10, "exercise": "curl"}]}
```

```bash
python evaluate.py data/                    # full report
python evaluate.py data/ --tune             # coordinate-descent threshold tuning
python evaluate.py data/ --csv-out err.csv  # per-set results
python evaluate.py --make-synthetic data/   # generate a sample dataset
```

Reported metrics: set-detection precision / recall / F1 (IoU matching); rep MAE, MAPE, % exact,
% within ±1, bias and Lin's CCC, overall and per exercise (exercises above 10 % MAPE are flagged);
exercise-ID accuracy and confusions when a classifier is supplied. Tune on one group of recordings and
confirm on a different group.

## Watch app (Zepp OS)

See `zepp-app/README.md` for setup. In short:

- Target: Amazfit Active Edge. The motion-sensor APIs need Zepp OS API level 3.0 or higher, so update
  the watch firmware first.
- `zepp-app/lib/wristlift.js` is a dependency-free port of the algorithm.
- `zepp-app/page/index.js` reads the sensors, resamples to 25 Hz and shows live reps, set count and a
  rest timer, plus the measured sensor rate for diagnostics.

## Status

| Component | State |
|---|---|
| Python set detection and rep counting | Tested on synthetic data only (curl-style and swing-style motion, 1.6-5 s rep periods, with/without gyro, 25 and 50 Hz, noise and tempo jitter; idle, fidgeting and walking-speed arm swing rejected) |
| JavaScript port | Matches the Python results on 16 of 16 synthetic cases when run in Node |
| `evaluate.py` | Verified on synthetic data |
| Zepp OS page (`page/index.js`) | Written, **not run on a watch** |
| Exercise recognition | Code only, no trained model |

Real-world accuracy has not been measured yet. Thresholds were set on synthetic signals and need tuning
on real recordings.

## Known limitations

- Exercises with little wrist movement (squats, lunges, deadlifts, push-ups, planks, dips) are hard to
  count from a wrist sensor.
- One repeated movement per set; mixed movements inside a set are not handled.
- Arm swing in the 0.2-1.3 Hz band can look like a set; faster motion such as walking is rejected.
- The watch app runs in the foreground with the screen kept awake. Background operation is not built.

## Roadmap

1. On-watch raw-data recorder and export, to collect labeled real sessions.
2. Tune thresholds with `evaluate.py` on real data.
3. Learned set detector and rep-boundary model to replace the threshold rules where they fail.
4. Trained exercise classifier (public datasets such as MM-Fit and RecGym plus your own recordings).
5. Per-user correction of rep counts used as fine-tuning labels.

## References

- Morris et al., *RecoFit: Using a Wearable Sensor to Find, Recognize, and Count Repetitive Exercises*, CHI 2014.
- Zepp OS developer documentation: https://docs.zepp.com
