Claude
Evaluate · PY
"""
evaluate.py - measure wristlift accuracy against your labeled recordings, and tune thresholds.
 
DATA LAYOUT  (one folder)
    session01.csv     columns: t,ax,ay,az[,gx,gy,gz]   (t in seconds; header row optional)
    session01.json    {"sets": [{"start": 12.4, "end": 41.0, "reps": 10, "exercise": "curl"}, ...]}
                      start/end = seconds in the CSV's time base, reps = your manual count.
 
USAGE
    python evaluate.py data/                      # report
    python evaluate.py data/ --tune               # tune thresholds (coordinate descent), print best Config
    python evaluate.py data/ --csv-out errors.csv # per-set results for plotting
    python evaluate.py --make-synthetic data/     # write a synthetic dataset to try the tool
 
METRICS
    Set detection : precision, recall, F1  (a detected set matches a true set if IoU >= --iou)
    Rep counting  : MAE, MAPE, % exact, % within +-1, bias, Lin's CCC   (matched sets only)
    Per exercise  : the same rep metrics grouped by the "exercise" label
    Exercise ID   : accuracy of the predicted label, only if --exercise-model is given (joblib)
Garmin-style acceptance: MAPE <= 10 % per exercise is a reasonable bar.
Always evaluate on recordings from people / sessions NOT used for tuning or training.
"""
from __future__ import annotations
 
import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List
 
import numpy as np
 
from wristlift import Config, WorkoutDetector, detect_sets, load_csv
 
 
# --------------------------------------------------------------------------- loading
def load_dataset(folder: str, fs: float):
    recs = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".csv"):
            continue
        js = os.path.join(folder, fn[:-4] + ".json")
        if not os.path.exists(js):
            print(f"skip {fn}: no matching .json")
            continue
        acc, gyro = load_csv(os.path.join(folder, fn), fs)
        truth = json.load(open(js))["sets"]
        recs.append({"name": fn[:-4], "acc": acc, "gyro": gyro, "truth": truth})
    if not recs:
        raise SystemExit(f"no usable recordings in {folder}")
    return recs
 
 
# --------------------------------------------------------------------------- matching + metrics
def iou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0
 
 
def match_sets(truth: List[dict], pred: List[dict], min_iou: float):
    """Greedy one-to-one matching by IoU. Returns (pairs, unmatched_truth_idx, unmatched_pred_idx)."""
    cands = sorted(((iou(t["start"], t["end"], p["start"], p["end"]), i, j)
                    for i, t in enumerate(truth) for j, p in enumerate(pred)), reverse=True)
    ut, up, pairs = set(range(len(truth))), set(range(len(pred))), []
    for v, i, j in cands:
        if v < min_iou:
            break
        if i in ut and j in up:
            pairs.append((i, j, v))
            ut.discard(i)
            up.discard(j)
    return pairs, sorted(ut), sorted(up)
 
 
def lin_ccc(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * cov / (x.var() + y.var() + (x.mean() - y.mean()) ** 2 + 1e-12))
 
 
def rep_metrics(true_r, pred_r) -> Dict[str, float]:
    t, p = np.asarray(true_r, float), np.asarray(pred_r, float)
    if not len(t):
        return {}
    err = p - t
    return {"n": len(t), "MAE": float(np.abs(err).mean()),
            "MAPE%": float(100 * np.mean(np.abs(err) / np.maximum(t, 1))),
            "exact%": float(100 * np.mean(err == 0)), "within1%": float(100 * np.mean(np.abs(err) <= 1)),
            "bias": float(err.mean()), "CCC": lin_ccc(t, p)}
 
 
def run_eval(recs, cfg: Config, min_iou: float, exercise_model=None):
    tp = fp = fn = 0
    rows = []
    for r in recs:
        det = WorkoutDetector(cfg, exercise_model=exercise_model)
        ev = []
        step = int(cfg.fs)
        for i in range(0, len(r["acc"]), step):
            ev += det.push(r["acc"][i:i + step], None if r["gyro"] is None else r["gyro"][i:i + step])
        ev += det.flush()
        pred = [e for e in ev if e["type"] == "set_end"]
        pairs, miss, extra = match_sets(r["truth"], pred, min_iou)
        tp, fn, fp = tp + len(pairs), fn + len(miss), fp + len(extra)
        for i, j, v in pairs:
            rows.append({"recording": r["name"], "exercise": r["truth"][i].get("exercise", "?"),
                         "true_reps": r["truth"][i]["reps"], "pred_reps": pred[j]["reps"],
                         "iou": round(v, 2), "pred_exercise": pred[j].get("exercise")})
        for i in miss:
            rows.append({"recording": r["name"], "exercise": r["truth"][i].get("exercise", "?"),
                         "true_reps": r["truth"][i]["reps"], "pred_reps": None, "iou": 0, "pred_exercise": None})
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    matched = [x for x in rows if x["pred_reps"] is not None]
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "rows": rows, "reps": rep_metrics([x["true_reps"] for x in matched],
                                              [x["pred_reps"] for x in matched])}
 
 
# --------------------------------------------------------------------------- reporting
def print_report(res):
    print("\n=== SET DETECTION ===")
    print(f"true sets matched {res['tp']}, missed {res['fn']}, false sets {res['fp']}")
    print(f"precision {res['precision']:.3f}  recall {res['recall']:.3f}  F1 {res['f1']:.3f}")
 
    def fmt(m):
        return (f"n={m['n']:<4} MAE={m['MAE']:.2f}  MAPE={m['MAPE%']:.1f}%  exact={m['exact%']:.0f}%  "
                f"+-1={m['within1%']:.0f}%  bias={m['bias']:+.2f}  CCC={m['CCC']:.3f}")
 
    print("\n=== REP COUNTING (matched sets) ===")
    print("overall      ", fmt(res["reps"]) if res["reps"] else "no matched sets")
    by = defaultdict(list)
    for x in res["rows"]:
        if x["pred_reps"] is not None:
            by[x["exercise"]].append(x)
    if len(by) > 1:
        print("\nper exercise:")
        for ex, xs in sorted(by.items()):
            m = rep_metrics([x["true_reps"] for x in xs], [x["pred_reps"] for x in xs])
            flag = "" if m["MAPE%"] <= 10 else "   <-- above 10% target"
            print(f"  {ex:<14}", fmt(m), flag)
 
    labeled = [x for x in res["rows"] if x["pred_exercise"] is not None]
    if labeled:
        acc = np.mean([x["exercise"] == x["pred_exercise"] for x in labeled])
        print(f"\n=== EXERCISE ID ===\naccuracy {100 * acc:.1f}% over {len(labeled)} sets")
        conf = defaultdict(int)
        for x in labeled:
            if x["exercise"] != x["pred_exercise"]:
                conf[(x["exercise"], x["pred_exercise"])] += 1
        for (a, b), n in sorted(conf.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  true {a} -> predicted {b}: {n}")
 
    worst = sorted((x for x in res["rows"] if x["pred_reps"] is not None),
                   key=lambda x: -abs(x["pred_reps"] - x["true_reps"]))[:5]
    if worst and abs(worst[0]["pred_reps"] - worst[0]["true_reps"]) > 0:
        print("\nworst rep errors (look at these recordings first):")
        for x in worst:
            print(f"  {x['recording']:<20} {x['exercise']:<12} true {x['true_reps']:>3}  pred {x['pred_reps']:>3}")
    missed = [x for x in res["rows"] if x["pred_reps"] is None]
    if missed:
        print(f"\nmissed sets ({len(missed)}):",
              ", ".join(f"{x['recording']}/{x['exercise']}" for x in missed[:8]))
 
 
# --------------------------------------------------------------------------- tuning
def tune(recs, base: Config, min_iou: float, passes: int = 2):
    """Coordinate descent: sweep one parameter at a time, keep the best, repeat.
    (~30 evaluations instead of a 243-point full grid.)"""
    grid = {"min_energy": [0.4, 0.7, 1.0], "min_ac": [0.25, 0.35, 0.5],
            "min_conc": [0.25, 0.35, 0.5], "prom_frac": [0.15, 0.25, 0.35], "end_idle_s": [5.0, 7.0, 10.0]}
 
    def score(cfg):
        r = run_eval(recs, cfg, min_iou)
        mape = r["reps"].get("MAPE%", 100.0) if r["reps"] else 100.0
        return r["f1"] - mape / 100.0, r, mape          # reward set detection, penalise rep error
 
    best = base
    best_score, best_res, mape = score(best)
    print(f"start: score {best_score:.3f}  F1 {best_res['f1']:.3f}  MAPE {mape:.1f}%")
    for p in range(passes):
        improved = False
        for key, values in grid.items():
            for v in values:
                if v == getattr(best, key):
                    continue
                cand = replace(best, **{key: v})
                sc, res, mp = score(cand)
                if sc > best_score + 1e-6:
                    best, best_score, best_res, improved = cand, sc, res, True
                    print(f"[pass {p + 1}] {key}={v}  score {sc:.3f}  F1 {res['f1']:.3f}  MAPE {mp:.1f}%")
        if not improved:
            break
    keys = list(grid)
    print("\nBEST CONFIG (paste into Config(...)):")
    print("Config(" + ", ".join(f"{k}={getattr(best, k)}" for k in keys) + f", fs={base.fs})")
    print("\nREMINDER: confirm on held-out recordings; numbers on the tuning set are optimistic.")
    print_report(best_res)
 
 
# --------------------------------------------------------------------------- synthetic dataset
def make_synthetic(folder: str, fs: int = 50):
    from wristlift import synth
    os.makedirs(folder, exist_ok=True)
    plans = {
        "curl": [("idle", 10), ("set", 10, 3.0, "curl"), ("idle", 25), ("set", 12, 2.4, "curl"),
                 ("idle", 20), ("set", 8, 3.5, "curl"), ("idle", 12)],
        "swing": [("idle", 9), ("set", 6, 2.0), ("idle", 20), ("set", 14, 2.8), ("idle", 12)],
    }
    for k, (name, plan) in enumerate(plans.items()):
        for seed in range(3):
            acc, gyro, _ = synth(fs, seed + 10 * k, plan)
            t = np.arange(len(acc)) / fs
            with open(os.path.join(folder, f"{name}_{seed}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz"])
                for i in range(len(acc)):
                    w.writerow([f"{t[i]:.3f}", *map(lambda v: f"{v:.5f}", acc[i]), *map(lambda v: f"{v:.5f}", gyro[i])])
            sets, cur = [], 0.0
            for p in plan:
                dur = p[1] if p[0] == "idle" else p[1] * p[2]
                if p[0] == "set":
                    sets.append({"start": round(cur, 2), "end": round(cur + dur, 2), "reps": p[1],
                                 "exercise": name})
                cur += dur
            json.dump({"sets": sets}, open(os.path.join(folder, f"{name}_{seed}.json"), "w"))
    print(f"wrote synthetic dataset to {folder}/")
 
 
# --------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--fs", type=float, default=50.0)
    ap.add_argument("--iou", type=float, default=0.3, help="min IoU to match a detected set to a true set")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--csv-out", help="write per-set results to this CSV")
    ap.add_argument("--exercise-model", help="joblib file holding a fitted ExerciseClassifier")
    ap.add_argument("--make-synthetic", metavar="DIR")
    a = ap.parse_args()
 
    if a.make_synthetic:
        make_synthetic(a.make_synthetic, int(a.fs))
        raise SystemExit
    if not a.folder:
        ap.error("folder required")
    recs = load_dataset(a.folder, a.fs)
    cfg = Config(fs=a.fs)
    if a.tune:
        tune(recs, cfg, a.iou)
    else:
        model = None
        if a.exercise_model:
            import joblib
            model = joblib.load(a.exercise_model)
        res = run_eval(recs, cfg, a.iou, model)
        print_report(res)
        if a.csv_out:
            with open(a.csv_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(res["rows"][0]))
                w.writeheader()
                w.writerows(res["rows"])
            print(f"\nper-set results written to {a.csv_out}")
 
