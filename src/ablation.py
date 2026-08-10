"""
Group ablation: does each block of new features actually earn its place?

Run as a script rather than a notebook cell - each group is a full LightGBM
fit, and an in-notebook version gives no progress visibility and loses
everything on a timeout.

Correlation is NOT the test. This project has been misled by it three times
(relative_size looked weak at 0.25 and cut error 20.1%->18.0%; coastal
distance looked dead at -0.07 and became 13.6% of what the tree uses).
Fitting with and without, and measuring held-out error, is the test.

Usage:
    venv/Scripts/python.exe src/ablation.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_INTERIM, DATA_PROCESSED

import numpy as np
import pandas as pd
import lightgbm as lgb

TRAIN_END = pd.Timestamp("2023-07-01")
VALID_END = pd.Timestamp("2024-07-01")   # test set starts here, unchanged since v0

# --- feature groups -------------------------------------------------------
BASE_NUM = ["log_area", "log_relsize", "dist_to_beach_km", "dist_to_marine_lake_m",
            "lat", "lon", "number_habitable_rooms", "number_heated_rooms"]
BASE_CAT = ["built_form_clean", "construction_age_band", "tenure", "duration"]

TYPE_CAT = ["dwelling_type"]                       # #2

HIST_NUM = ["log_prev_price_adj", "years_since_prev_sale", "prev_sale_count",
            "area_change", "sap_change", "extension_change", "glazing_change",
            "heating_upgraded", "short_hold", "work_done_since_prev",
            "likely_flip", "uprn_n_addresses"]     # #4 #6 #7

LOC_NUM = ["local_ppsqm", "local_sales_count", "dist_to_station_m",
           "dist_to_park_m", "dist_to_good_school_m"]   # #10 #14 #15 #17

GROUPS = [
    ("v5 baseline (as before)",        BASE_NUM,                        BASE_CAT),
    ("+ dwelling type",                BASE_NUM,                        BASE_CAT + TYPE_CAT),
    ("+ sale history & renovation",    BASE_NUM + HIST_NUM,             BASE_CAT + TYPE_CAT),
    ("+ location & amenities (v6)",    BASE_NUM + HIST_NUM + LOC_NUM,   BASE_CAT + TYPE_CAT),
]

# Leave-one-group-out from the full set, to catch a group that actively hurts.
LOGO = [
    ("v6 minus dwelling type",         BASE_NUM + HIST_NUM + LOC_NUM,   BASE_CAT),
    ("v6 minus sale history",          BASE_NUM + LOC_NUM,              BASE_CAT + TYPE_CAT),
    ("v6 minus location",              BASE_NUM + HIST_NUM,             BASE_CAT + TYPE_CAT),
]

PARAMS = dict(objective="regression", n_estimators=3000, learning_rate=0.033,
              num_leaves=180, min_child_samples=10, subsample=0.944,
              subsample_freq=1, colsample_bytree=0.60, reg_alpha=0.036,
              reg_lambda=1.77, random_state=42, verbose=-1, n_jobs=-1)


def load():
    df = pd.read_parquet(DATA_INTERIM / "features.parquet")
    df["log_price_adj"] = np.log(df["price_adjusted"])
    df["log_area"] = np.log(df["total_floor_area"])
    df["log_relsize"] = np.log(df["relative_size"])
    df["dist_to_beach_km"] = df["dist_to_beach_m"] / 1000
    for c in ["number_habitable_rooms", "number_heated_rooms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in BASE_CAT + TYPE_CAT:
        df[c] = df[c].astype("category")
    df["outcode"] = df["postcode"].str.split(" ").str[0]
    return df.dropna(subset=["log_price_adj", "date_of_transfer", "log_area"])


def score(pred_log, actual_log):
    pred, actual = np.exp(pred_log), np.exp(actual_log)
    ape = (pred - actual).abs() / actual
    ss_res = ((actual_log - pred_log) ** 2).sum()
    ss_tot = ((actual_log - actual_log.mean()) ** 2).sum()
    return {"r2": 1 - ss_res / ss_tot,
            "mdape": float(ape.median()),
            "ppe10": float((ape <= 0.10).mean())}


def run(df, label, num, cat):
    feats = num + cat
    # Monotonic only on floor area - the one relationship we can assert.
    mono = [1 if f == "log_area" else 0 for f in feats]

    tr = df[df["date_of_transfer"] < TRAIN_END]
    va = df[(df["date_of_transfer"] >= TRAIN_END) & (df["date_of_transfer"] < VALID_END)]
    te = df[df["date_of_transfer"] >= VALID_END]

    m = lgb.LGBMRegressor(**PARAMS, monotone_constraints=mono)
    m.fit(tr[feats], tr["log_price_adj"],
          eval_set=[(va[feats], va["log_price_adj"])], eval_metric="l2",
          callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)])

    pred = pd.Series(m.predict(te[feats]), index=te.index)
    overall = score(pred, te["log_price_adj"])

    crosby = te["outcode"] == "L23"
    cro = score(pred[crosby], te.loc[crosby, "log_price_adj"])

    print(f"  {label:32s} R2 {overall['r2']:.3f}  MdAPE {overall['mdape']:.1%}  "
          f"PPE10 {overall['ppe10']:.1%}   |  Crosby MdAPE {cro['mdape']:.1%} "
          f"PPE10 {cro['ppe10']:.1%}   ({m.best_iteration_} trees)", flush=True)
    return {"label": label, "n_features": len(feats), "trees": m.best_iteration_,
            **{f"overall_{k}": v for k, v in overall.items()},
            **{f"crosby_{k}": v for k, v in cro.items()}}


def main():
    df = load()
    te_n = (df["date_of_transfer"] >= VALID_END).sum()
    print(f"{len(df):,} rows | test set {te_n:,} from {VALID_END.date()}\n")

    print("CUMULATIVE - each group added on top of the last:")
    results = [run(df, *g) for g in GROUPS]

    print("\nLEAVE-ONE-OUT - drop one group from the full set:")
    results += [run(df, *g) for g in LOGO]

    out = pd.DataFrame(results)
    out.to_csv(DATA_PROCESSED / "ablation_results.csv", index=False)
    print(f"\nSaved to {DATA_PROCESSED / 'ablation_results.csv'}")

    base, full = results[0], results[3]
    print(f"\nBaseline -> v6:  MdAPE {base['overall_mdape']:.1%} -> {full['overall_mdape']:.1%}"
          f"   Crosby {base['crosby_mdape']:.1%} -> {full['crosby_mdape']:.1%}")


if __name__ == "__main__":
    main()
