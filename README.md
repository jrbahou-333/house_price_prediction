# Crosby house price model

Predicting UK house prices in Sefton (Merseyside), to answer one practical
question: **for a property at a given asking price, is that price above or
below what the market says it's worth?**

Built as a learning project — the notebooks explain the reasoning and the ML
concepts as they go, not just the code.

## The pipeline

Notebooks run in order. Each stage writes to `data/interim/` so the next one
can pick up from there.

| Notebook | Does |
|---|---|
| `00_setup_and_download.ipynb` | Downloads Price Paid, EPC, HPI, OSM data |
| `01_preprocessing.ipynb` | Cleans Price Paid, cleans EPC, joins them by address |
| `02_eda.ipynb` | Explores what actually correlates with price — informs the features below |
| `03_features.ipynb` | Builds every model feature (inflation adjustment, geocoding, relative size, dwelling type, sale history, renovation detection, local market level, amenity distances) |
| `04_model.ipynb` | Fits and compares model versions on a temporal train/test split |

Longer-running steps are scripts rather than notebook cells — they give
progress output, survive interruption, and don't lose everything on a
timeout (an early in-notebook tuning run hit a 4-hour limit and lost all of
it):

| Script | Does |
|---|---|
| `src/tune.py` | Optuna hyperparameter search, resumable via SQLite |
| `src/ablation.py` | Group ablation — is each block of features actually earning its place? |
| `src/calibrate.py` | Prediction intervals (conformalized quantile regression) and the confidence flag |

## Data sources

All free and open (OGL), no paid APIs:

- **[HM Land Registry Price Paid](https://www.gov.uk/guidance/about-the-price-paid-data)** — every residential sale in England & Wales. Gives the target variable (price).
- **[EPC domestic certificates](https://get-energy-performance-data.communities.gov.uk/)** — floor area, rooms, property type, age band. Requires a free GOV.UK One Login; download filtered to a single council.
- **[UK House Price Index](https://landregistry.data.gov.uk/app/ukhpi)** — monthly index used to restate historic sales in today's money.
- **[UBDC Price Paid → UPRN lookup](https://data.ubdc.ac.uk/)** — published address-matching lookup (free account). Covers 1995–Jan 2022.
- **[postcodes.io](https://postcodes.io/)** — postcode → coordinates, no key needed.
- **[Geofabrik OSM extract](https://download.geofabrik.de/europe/great-britain/england/merseyside.html)** — beaches, Marine Lake, for coastal distance.

## Results

Five local authorities (all of Merseyside), **287,946 joined transactions**.
Measured on **held-out future sales** (train to mid-2023, validate to
mid-2024, test after) — honest out-of-sample numbers, not scores on data the
model saw.

| Model version | Test R² | MdAPE | PPE10 |
|---|---|---|---|
| v0: floor area only, no inflation adjustment | 0.03 | 35.0% | 11.9% |
| v1: + inflation adjustment | 0.47 | 23.0% | 21.5% |
| v2: + relative size (vs 15 nearest neighbours) | 0.54 | 21.0% | 23.8% |
| v3: + house type + coastal distance (linear) | 0.65 | 16.8% | 31.4% |
| v4b: LightGBM, extended features | 0.83 | 11.1% | 45.9% |
| v5: LightGBM tuned (Optuna) | 0.84 | 10.6% | 47.4% |
| **v6: + condition, sale history, amenities** | **0.86** | **9.3%** | **52.9%** |

**MdAPE** = median absolute percentage error — typical error size. **PPE10** =
share of predictions within 10% of the true price (the standard automated-
valuation benchmark).

**On Crosby specifically** (742 test properties): **8.0% MdAPE / 59.2%
PPE10**, better than the Merseyside average since Crosby is more homogeneous
than Liverpool or Knowsley. Narrowing further to a **£250–350k
semi-detached house** — the most common purchase profile — gives **5.9%
MdAPE / 68.2% PPE10**.

### The confidence flag matters more than the average

Error is highly concentrated: the best half of predictions average **4.3%**,
the worst tenth **37.6%**, and that worst tenth carries **38% of all error**.
A single accuracy figure hides which case you are in, so every prediction
carries a calibrated interval and a flag.

| Flag | Share of Crosby | MdAPE | PPE10 |
|---|---|---|---|
| 🟢 Green | 33.4% | **5.1%** | 73.8% |
| 🟡 Amber | 56.7% | 8.6% | 55.8% |
| 🔴 Red | 9.8% | 16.1% | 37.0% |

Intervals come from **conformalized quantile regression** — plain conformal
prediction gives every property the same width, which tells you nothing.
Empirical coverage is **79.8% against an 80% target**, verified on unseen
data.

⚠️ Coverage is uneven at the extremes: 75.3% below £150k and 76.3% above
£400k against 82.7% in the middle. The intervals are slightly too narrow at
both ends.

### What each feature group was worth

Measured by ablation — fitting with and without, on the same held-out sales.
Correlation is deliberately *not* used to decide this; it has been
misleading three times in this project.

| Group | Cost of removing it (Crosby MdAPE) |
|---|---|
| Sale history, renovation, flips | **+1.4pp** |
| Dwelling type (house/flat/bungalow) | +0.7pp |
| Local market, stations, parks, schools | +0.4pp |

No group was dead weight. Baseline to v6 on Crosby: **9.9% → 8.0%**.

### How wide should the training area be?

Tested directly: three models, tuned identically, all scored on the same
746 Crosby properties.

| Trained on | Test R² | MdAPE | PPE10 |
|---|---|---|---|
| All six authorities (226k sales) | 0.815 | **9.1%** | 52.8% |
| Sefton only (44k sales) | **0.824** | 9.3% | **54.2%** |
| Crosby only (5k sales) | 0.817 | 10.2% | 49.5% |

(Measured before West Lancashire was dropped, so "six authorities" here is
the then-current scope. The conclusion — train wider than the target area —
is unchanged.)

Genuinely close between the two wider options — six authorities edge the
median error, Sefton-only edges R² and PPE10. Crosby-only is clearly worst,
so training on more than the target area does help. An earlier run
suggested widening *hurt*, but that turned out to be undertraining: the
untuned models were hitting their tree ceiling, and the effect disappeared
once tuned.

### Findings worth noting

- **The address join reached 95.3%** (vs ~93% in published work) and held
  at exactly that when widened from one authority to six, with Route B
  precision improving 96.1% → 97.4%. Flats match worst (75%), and
  post-2022 sales are weaker because the UBDC lookup stops at Jan 2022.
- **Inflation adjustment mattered more than any single feature.** Prices
  rose ~63% over the period; uncorrected, the model scored ≈0 on future
  sales despite looking fine in training. It must be applied *per
  authority* — Liverpool's index is 113.7 against Knowsley's 107.4.
- **Location dominates once the area widens.** Latitude, longitude and the
  two coastal distances together account for ~55% of what the model uses;
  floor area drops to 11%. The linear model couldn't exploit this, which is
  most of why the tree model beats it.
- **EPC energy ratings barely predict price** — but room counts and house
  type predict it strongly.
- **`relative_size` helped, but not the way the hypothesis predicted.** It
  was meant to rescue prediction on uniform-size streets; it helped most on
  *varied* streets instead.
- **Tuning bought less than the data and features did** — 11.1% → 10.6%,
  with only 0.5pp spread across all 20 trials.
- **A leakage assertion caught three real bugs**, all of which would have
  made the model look *better* than it is by feeding it a near-perfect
  predictor derived from the answer: Land Registry recording one sale twice
  under two IDs; the address matcher collapsing several flats in a building
  onto one UPRN (so a neighbour's sale became "this property's" history);
  and a dedupe that normalised postcodes one line *after* keying on them.
  None would have raised an error. Sale history is now keyed on the address
  rather than the UPRN.
- **Bias correction did not work and was dropped.** An isotonic correction
  fitted on held-out data made both tails *worse* (<£150k: +4.9% → +6.0%;
  £400k+: −1.8% → −4.1%) — too few extreme-priced rows in the calibration
  slice to fit the curve there, so it overfit the middle.

## Known limitations

- **Cosmetic condition is still invisible.** The renovation features detect
  extensions and efficiency upgrades, not kitchens, bathrooms or damp.
- **13% of matched sales have no EPC**, so no floor area. Disproportionately
  long-held owner-occupied homes — missing from both datasets at once.
- **Condition is unobserved.** No free UK equivalent of a build-quality
  grade exists; it's the single biggest missing predictor.
- **No bathroom count** — a strong predictor in comparable work, absent from
  every free UK source.
- The tuned model still trains close to its tree ceiling, so it may not be
  fully converged.

## Setup

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
venv/Scripts/python.exe -m ipykernel install --user --name house_price_prediction
```

Then run `00_setup_and_download.ipynb` — it lists the two manual steps
(GOV.UK One Login for EPC, free UBDC account for the UPRN lookup).
