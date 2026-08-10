"""
Turn a point prediction into an honest one: bias correction, calibrated
prediction intervals, and a confidence flag.

Three pieces:

1. BIAS CORRECTION - the model hedges toward the average, over-predicting
   cheap properties and under-predicting expensive ones. An isotonic
   (monotonic) correction fitted on held-out data straightens that out
   without changing the ordering of predictions.

2. CONFORMALIZED QUANTILE REGRESSION (CQR) - plain conformal prediction
   gives every property the SAME interval width, which is useless for
   telling you which predictions to trust. CQR trains explicit 10th/90th
   percentile models, then corrects them so coverage is guaranteed. The
   width then varies per property, which is what makes a confidence flag
   meaningful.

3. CONFIDENCE FLAG - green/amber/red from the interval width relative to
   the prediction.

Calibration data is split in two so bias correction and interval
calibration never share rows. The TEST set is untouched until final
scoring.

Usage:
    venv/Scripts/python.exe src/calibrate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_PROCESSED

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

import src.ablation as ab

ALPHA = 0.20            # 80% prediction interval
SEED = 42


def fit_point(tr, va, feats, mono):
    m = lgb.LGBMRegressor(**ab.PARAMS, monotone_constraints=mono)
    m.fit(tr[feats], tr["log_price_adj"],
          eval_set=[(va[feats], va["log_price_adj"])], eval_metric="l2",
          callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)])
    return m


def fit_quantile(tr, va, feats, q):
    # No monotone_constraints here: LightGBM rejects them with the quantile
    # objective. That's acceptable - the constraint exists so the *point
    # estimate* never says a bigger house is worth less, and that model
    # still carries it. These two only supply interval bounds, which are
    # clamped around the point estimate below anyway.
    p = {**ab.PARAMS, "objective": "quantile", "alpha": q}
    m = lgb.LGBMRegressor(**p)
    m.fit(tr[feats], tr["log_price_adj"],
          eval_set=[(va[feats], va["log_price_adj"])], eval_metric="quantile",
          callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)])
    return m


def metrics(pred, actual):
    ape = (pred - actual).abs() / actual
    return {"mdape": float(ape.median()),
            "ppe10": float((ape <= 0.10).mean()),
            "bias": float(((pred - actual) / actual).median())}


def main():
    df = ab.load()
    feats = ab.BASE_NUM + ab.HIST_NUM + ab.LOC_NUM + ab.BASE_CAT + ab.TYPE_CAT
    mono = [1 if f == "log_area" else 0 for f in feats]

    tr = df[df["date_of_transfer"] < ab.TRAIN_END]
    va = df[(df["date_of_transfer"] >= ab.TRAIN_END) & (df["date_of_transfer"] < ab.VALID_END)]
    te = df[df["date_of_transfer"] >= ab.VALID_END].copy()

    # Split calibration data so bias correction and interval calibration
    # never see the same rows.
    rng = np.random.default_rng(SEED)
    mask = rng.random(len(va)) < 0.5
    cal_bias, cal_int = va[mask], va[~mask]
    print(f"train {len(tr):,} | calib-bias {len(cal_bias):,} | "
          f"calib-interval {len(cal_int):,} | test {len(te):,}\n")

    print("Fitting point model...", flush=True)
    point = fit_point(tr, va, feats, mono)
    print("Fitting 10th percentile model...", flush=True)
    q_lo = fit_quantile(tr, va, feats, ALPHA / 2)
    print("Fitting 90th percentile model...", flush=True)
    q_hi = fit_quantile(tr, va, feats, 1 - ALPHA / 2)
    print()

    # ---------------------------------------------------------- 1. bias
    raw_cal = point.predict(cal_bias[feats])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, cal_bias["log_price_adj"])

    raw_te = point.predict(te[feats])
    te["pred_raw"] = np.exp(raw_te)
    te["pred"] = np.exp(iso.predict(raw_te))
    te["actual"] = np.exp(te["log_price_adj"])

    before = metrics(te["pred_raw"], te["actual"])
    after = metrics(te["pred"], te["actual"])
    print("=== 1. Bias correction ===")
    print(f"  before: MdAPE {before['mdape']:.1%}  PPE10 {before['ppe10']:.1%}  bias {before['bias']:+.1%}")
    print(f"  after : MdAPE {after['mdape']:.1%}  PPE10 {after['ppe10']:.1%}  bias {after['bias']:+.1%}")

    te["band"] = pd.cut(te["actual"], [0, 150e3, 250e3, 400e3, 1e9],
                        labels=["<150k", "150-250k", "250-400k", "400k+"])
    print("\n  Signed bias by price band (target: within +-2%):")
    for lbl, g in te.groupby("band", observed=True):
        b0 = ((g["pred_raw"] - g["actual"]) / g["actual"]).median()
        b1 = ((g["pred"] - g["actual"]) / g["actual"]).median()
        print(f"    {str(lbl):10s} n={len(g):>6,}   before {b0:+6.1%}   after {b1:+6.1%}")

    # ------------------------------------------------- 2. CQR intervals
    lo_cal = q_lo.predict(cal_int[feats])
    hi_cal = q_hi.predict(cal_int[feats])
    y_cal = cal_int["log_price_adj"].to_numpy()

    # Conformity score: how far outside the predicted band did truth fall?
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    q_hat = np.sort(scores)[min(k, n) - 1]

    te["lo"] = np.exp(q_lo.predict(te[feats]) - q_hat)
    te["hi"] = np.exp(q_hi.predict(te[feats]) + q_hat)
    # An interval that excludes its own point estimate would be incoherent.
    te["lo"] = np.minimum(te["lo"], te["pred"])
    te["hi"] = np.maximum(te["hi"], te["pred"])

    covered = (te["actual"] >= te["lo"]) & (te["actual"] <= te["hi"])
    print(f"\n=== 2. Prediction intervals ({100*(1-ALPHA):.0f}% target) ===")
    print(f"  empirical coverage overall: {covered.mean():.1%}   (target {1-ALPHA:.0%}, ok 78-82%)")
    print("\n  Coverage by price band - the check that matters:")
    for lbl, g in te.groupby("band", observed=True):
        c = ((g["actual"] >= g["lo"]) & (g["actual"] <= g["hi"])).mean()
        w = ((g["hi"] - g["lo"]) / g["pred"] / 2).median()
        print(f"    {str(lbl):10s} n={len(g):>6,}   coverage {c:.1%}   median half-width +-{w:.1%}")

    # ------------------------------------------------ 3. confidence flag
    te["half_width_pct"] = (te["hi"] - te["lo"]) / te["pred"] / 2
    te["flag"] = pd.cut(te["half_width_pct"], [0, 0.15, 0.25, 9],
                        labels=["GREEN", "AMBER", "RED"])

    print("\n=== 3. Confidence flag ===")
    print(f"{'flag':7s} {'share':>7s} {'n':>7s} {'MdAPE':>7s} {'PPE10':>7s} {'coverage':>9s}")
    for lbl, g in te.groupby("flag", observed=True):
        c = ((g["actual"] >= g["lo"]) & (g["actual"] <= g["hi"])).mean()
        mm = metrics(g["pred"], g["actual"])
        print(f"{str(lbl):7s} {len(g)/len(te):>6.1%} {len(g):>7,} "
              f"{mm['mdape']:>6.1%} {mm['ppe10']:>6.1%} {c:>8.1%}")

    crosby = te[te["outcode"] == "L23"]
    print(f"\n  Crosby only (n={len(crosby):,}):")
    for lbl, g in crosby.groupby("flag", observed=True):
        if len(g) < 10:
            continue
        mm = metrics(g["pred"], g["actual"])
        print(f"    {str(lbl):7s} {len(g)/len(crosby):>6.1%} of Crosby   "
              f"MdAPE {mm['mdape']:.1%}   PPE10 {mm['ppe10']:.1%}")

    out = DATA_PROCESSED / "calibrated_test_predictions.parquet"
    te[["transaction_id", "postcode", "outcode", "actual", "pred", "lo", "hi",
        "half_width_pct", "flag", "band"]].to_parquet(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
