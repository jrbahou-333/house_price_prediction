"""
Value a single property: point estimate, calibrated interval, RAG flag,
SHAP breakdown and comparables.

Asking price is used ONLY for the verdict, never as a model input - feeding
it in would be circular, since the whole question is whether the asking
price is right.

Usage:
    venv/Scripts/python.exe src/predict.py "53" "L23 0TG" --asking 330000
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_INTERIM, DATA_RAW

import numpy as np
import pandas as pd
import lightgbm as lgb
import src.ablation as ab

ALPHA = 0.20
TODAY_INDEX_NOTE = "prices are modelled in today's money (HPI-adjusted), so the output is a present-day valuation"


def build_models(df, feats, mono):
    tr = df[df["date_of_transfer"] < ab.TRAIN_END]
    va = df[(df["date_of_transfer"] >= ab.TRAIN_END) & (df["date_of_transfer"] < ab.VALID_END)]
    cal = df[df["date_of_transfer"] >= ab.VALID_END]   # most recent data calibrates intervals

    point = lgb.LGBMRegressor(**ab.PARAMS, monotone_constraints=mono)
    point.fit(tr[feats], tr["log_price_adj"], eval_set=[(va[feats], va["log_price_adj"])],
              eval_metric="l2", callbacks=[lgb.early_stopping(75, verbose=False)])

    qs = {}
    for q in (ALPHA / 2, 1 - ALPHA / 2):
        m = lgb.LGBMRegressor(**{**ab.PARAMS, "objective": "quantile", "alpha": q})
        m.fit(tr[feats], tr["log_price_adj"], eval_set=[(va[feats], va["log_price_adj"])],
              eval_metric="quantile", callbacks=[lgb.early_stopping(75, verbose=False)])
        qs[q] = m

    # Conformal correction, calibrated on held-out recent sales.
    lo_c = qs[ALPHA / 2].predict(cal[feats])
    hi_c = qs[1 - ALPHA / 2].predict(cal[feats])
    y_c = cal["log_price_adj"].to_numpy()
    scores = np.maximum(lo_c - y_c, y_c - hi_c)
    k = int(np.ceil((len(scores) + 1) * (1 - ALPHA)))
    q_hat = np.sort(scores)[min(k, len(scores)) - 1]
    return point, qs, q_hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paon"); ap.add_argument("postcode")
    ap.add_argument("--asking", type=float, default=None)
    a = ap.parse_args()

    df = ab.load()
    feats = ab.BASE_NUM + ab.HIST_NUM + ab.LOC_NUM + ab.BASE_CAT + ab.TYPE_CAT
    mono = [1 if f == "log_area" else 0 for f in feats]

    full = pd.read_parquet(DATA_INTERIM / "features.parquet")
    hit = full[(full["paon"] == a.paon) & (full["postcode"] == a.postcode)]
    if hit.empty:
        sys.exit(f"No record for {a.paon}, {a.postcode}. Supply details manually.")
    prop = hit.sort_values("date_of_transfer").iloc[-1]

    # --- build a row describing the property AS IT WOULD SELL TODAY ---
    today = pd.Timestamp("2026-06-30")
    row = prop.copy()
    row["date_of_transfer"] = today
    row["log_prev_price_adj"] = np.log(prop["price_adjusted"])
    row["years_since_prev_sale"] = (today - prop["date_of_transfer"]).days / 365.25
    row["prev_sale_count"] = prop["prev_sale_count"] + 1
    row["short_hold"] = float(row["years_since_prev_sale"] < 2)
    for c in ["area_change", "sap_change", "extension_change", "glazing_change"]:
        row[c] = np.nan
    row["heating_upgraded"] = 0.0
    row["work_done_since_prev"] = np.nan
    row["likely_flip"] = 0.0

    # The feature row carries the EPC current at the LAST SALE. For a
    # valuation today we want the property as it is NOW - a newer
    # certificate may show an extension, a re-measurement, or upgrades.
    epc = pd.read_parquet(DATA_INTERIM / "epc_clean.parquet")
    mine = epc[epc["UPRN"] == prop["uprn_final"]].sort_values("inspection_date")
    epc_note = None
    if len(mine):
        cur = mine.iloc[-1]
        old_area = prop["total_floor_area"]
        if cur["inspection_date"] > prop["date_of_transfer"]:
            epc_note = (f"newer EPC dated {cur['inspection_date']:%d %b %Y} supersedes the "
                        f"one at the last sale ({old_area:.0f} -> {cur['total_floor_area']:.0f} m2)")
            row["total_floor_area"] = cur["total_floor_area"]
            for src, dst in [("number_habitable_rooms", "number_habitable_rooms"),
                             ("number_heated_rooms", "number_heated_rooms"),
                             ("construction_age_band", "construction_age_band")]:
                if pd.notna(cur[src]):
                    row[dst] = cur[src]
            # size relative to neighbours must move with the new area
            if pd.notna(prop["neighbourhood_mean_area"]):
                row["relative_size"] = cur["total_floor_area"] / prop["neighbourhood_mean_area"]
            # a newer certificate IS the renovation evidence the model looks for
            at_sale = mine[mine["inspection_date"] <= prop["date_of_transfer"]]
            if len(at_sale):
                row["area_change"] = cur["total_floor_area"] - at_sale.iloc[-1]["total_floor_area"]
                row["work_done_since_prev"] = float(row["area_change"] > 5)

    print(f"=== {a.paon} {prop['street']}, {a.postcode} ===")
    print(f"{row['total_floor_area']:.0f} m2  {row['built_form_clean']} {row['dwelling_type']}  "
          f"{row['number_habitable_rooms']} habitable rooms  {row['construction_age_band']}")
    print(f"Last sold {prop['date_of_transfer']:%d %b %Y} for GBP{prop['price']:,.0f} "
          f"(= GBP{prop['price_adjusted']:,.0f} in today's money)")
    if epc_note:
        print(f"NOTE: {epc_note}")
    print()

    # refresh the local market level to the latest available for this sector
    sector = f"{a.postcode.split()[0]} {a.postcode.split()[1][0]}"
    sec = full[full["pc_sector"] == sector].sort_values("date_of_transfer")
    if len(sec):
        row["local_ppsqm"] = sec["local_ppsqm"].dropna().iloc[-1]
        row["local_sales_count"] = sec["local_sales_count"].dropna().iloc[-1]

    X = pd.DataFrame([row])
    X["log_price_adj"] = 0.0
    X["log_area"] = np.log(X["total_floor_area"])
    X["log_relsize"] = np.log(X["relative_size"])
    X["dist_to_beach_km"] = X["dist_to_beach_m"] / 1000
    for c in ["number_habitable_rooms", "number_heated_rooms"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in ab.BASE_CAT + ab.TYPE_CAT:
        X[c] = pd.Categorical(X[c], categories=df[c].cat.categories)

    print("Fitting models...", flush=True)
    point, qs, q_hat = build_models(df, feats, mono)

    pred = float(np.exp(point.predict(X[feats])[0]))
    lo = float(np.exp(qs[ALPHA / 2].predict(X[feats])[0] - q_hat))
    hi = float(np.exp(qs[1 - ALPHA / 2].predict(X[feats])[0] + q_hat))
    lo, hi = min(lo, pred), max(hi, pred)
    half = (hi - lo) / pred / 2
    flag = "GREEN" if half <= 0.15 else ("AMBER" if half <= 0.25 else "RED")

    print()
    print(f"VALUATION      GBP{pred:,.0f}")
    print(f"80% interval   GBP{lo:,.0f} - GBP{hi:,.0f}   (+-{half:.1%})")
    print(f"CONFIDENCE     {flag}")
    if a.asking:
        d = (a.asking - pred) / pred
        verdict = ("ABOVE the model's range" if a.asking > hi else
                   "BELOW the model's range" if a.asking < lo else
                   "WITHIN the model's range")
        print(f"\nAsking GBP{a.asking:,.0f} is {d:+.1%} vs the estimate - {verdict}")

    # --- SHAP, in pounds ---
    import shap
    ex = shap.TreeExplainer(point)
    sv = ex.shap_values(X[feats])[0]
    base = ex.expected_value
    print("\n--- What drives this valuation ---")
    print(f"{'starting point (average property)':44s} GBP{np.exp(base):>9,.0f}")
    run = base
    for f, c in sorted(zip(feats, sv), key=lambda t: -abs(t[1]))[:8]:
        b4 = np.exp(run); run += c; print(f"{f:44s} {np.exp(run)-b4:>+10,.0f}")
    print(f"{'':44s} {'':>10}")
    print(f"{'PREDICTED':44s} GBP{np.exp(run):>9,.0f}")

    # --- comparables ---
    print("\n--- Nearest recent comparable sales (today's money) ---")
    comp = full[(full["pc_sector"] == sector)
                & (full["date_of_transfer"] >= "2024-01-01")
                & (full["built_form_clean"] == prop["built_form_clean"])
                & (full["dwelling_type"] == prop["dwelling_type"])].copy()
    comp["d"] = np.hypot(comp["lat"] - prop["lat"], comp["lon"] - prop["lon"])
    cols = ["date_of_transfer", "paon", "street", "total_floor_area", "price", "price_adjusted"]
    print(comp.nsmallest(10, "d")[cols].to_string(index=False,
          formatters={"price": "{:,.0f}".format, "price_adjusted": "{:,.0f}".format,
                      "date_of_transfer": lambda d: d.strftime("%b %Y")}))


if __name__ == "__main__":
    main()
