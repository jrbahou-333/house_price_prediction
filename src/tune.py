"""
Optuna hyperparameter search, run as a standalone script rather than a
notebook cell.

Why standalone: the search takes hours, and a notebook cell gives no
progress visibility (nbconvert buffers output) and loses everything if it
times out. This writes each trial to an SQLite study on disk, so the search
is resumable - rerun after an interruption and it picks up where it left
off rather than starting over.

Usage:
    venv/Scripts/python.exe src/tune.py [n_trials]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_INTERIM, DATA_PROCESSED

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
STUDY_DB = f"sqlite:///{(DATA_PROCESSED / 'optuna_study.db').as_posix()}"
STUDY_NAME = "lgbm_house_price"

TRAIN_END = pd.Timestamp("2023-07-01")
VALID_END = pd.Timestamp("2024-07-01")

NUMERIC = ["log_area", "log_relsize", "dist_to_beach_km", "dist_to_marine_lake_m",
           "lat", "lon", "number_habitable_rooms", "number_heated_rooms"]
CATEGORICAL = ["built_form_clean", "construction_age_band", "tenure", "duration"]
FEATURES = NUMERIC + CATEGORICAL

# Only floor area is constrained: "bigger is worth more" is the one
# relationship we can assert with confidence.
MONOTONE = [1] + [0] * (len(FEATURES) - 1)


def load():
    df = pd.read_parquet(DATA_INTERIM / "features.parquet")
    df["log_price_adj"] = np.log(df["price_adjusted"])
    df["log_area"] = np.log(df["total_floor_area"])
    df["log_relsize"] = np.log(df["relative_size"])
    df["dist_to_beach_km"] = df["dist_to_beach_m"] / 1000
    for c in ["number_habitable_rooms", "number_heated_rooms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    df = df.dropna(subset=["log_price_adj", "date_of_transfer", "log_area"])

    train = df[df["date_of_transfer"] < TRAIN_END]
    valid = df[(df["date_of_transfer"] >= TRAIN_END) & (df["date_of_transfer"] < VALID_END)]
    return df, train, valid


def main():
    df, train, valid = load()
    print(f"{len(df):,} rows | {len(train):,} train | {len(valid):,} validation", flush=True)

    def objective(trial):
        params = {
            "objective": "regression",
            "n_estimators": 3000,
            "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            # Capped at 255, not 512 - larger trees were the main cost driver
            # and showed no sign of being needed.
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq":    1,
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
            "monotone_constraints": MONOTONE,
            "n_jobs": -1,
        }

        t0 = time.time()
        m = lgb.LGBMRegressor(**params)
        m.fit(
            train[FEATURES], train["log_price_adj"],
            eval_set=[(valid[FEATURES], valid["log_price_adj"])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)],
        )
        pred = np.exp(m.predict(valid[FEATURES]))
        actual = np.exp(valid["log_price_adj"])
        mdape = float(((pred - actual).abs() / actual).median())

        print(f"  trial {trial.number:3d}: MdAPE {mdape:.2%}  "
              f"({m.best_iteration_} trees, {time.time() - t0:.0f}s)", flush=True)
        return mdape

    # SQLite storage makes the study resumable across interruptions.
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=STUDY_DB,
        study_name=STUDY_NAME,
        load_if_exists=True,
    )
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"{done} trials already complete; running {N_TRIALS} more\n", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nBest validation MdAPE: {study.best_value:.2%}")
    print("Best settings:")
    for k, v in study.best_params.items():
        print(f"  {k:20s} {v}")

    out = DATA_PROCESSED / "best_params.json"
    import json
    out.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
