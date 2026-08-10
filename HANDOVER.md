# Handover

## The goal

Given a house in Crosby at a given asking price, say whether that price is
above or below what the market says it's worth — with an honest signal of
how much to trust the answer.

Built as a learning project, so the notebooks explain the reasoning, not
just the code.

## Where it stands

**Working end-to-end on 287,946 real sales** across the five Merseyside
authorities, 2008–2026. Measured on held-out *future* sales.

| | Overall | Crosby | £250–350k Crosby semi |
|---|---|---|---|
| Median error | 9.3% | 8.0% | **5.9%** |
| Within 10% | 52.9% | 59.2% | 68.2% |

Started at 33.6%. The £250–350k semi is the most common purchase profile
and the model's strongest segment.

**Every prediction carries a confidence flag**, which matters more than the
average because error is very unevenly spread — the best half of
predictions average 4.3%, the worst tenth 37.6%, and that worst tenth
carries 38% of all error.

| Flag | Share of Crosby | Median error |
|---|---|---|
| 🟢 Green | 33.4% | **5.1%** |
| 🟡 Amber | 56.7% | 8.6% |
| 🔴 Red | 9.8% | 16.1% |

Intervals are calibrated: 79.8% empirical coverage against an 80% target.

## Pipeline

Notebooks run in order; long jobs are scripts so they survive interruption.

```
00_setup_and_download   downloads
01_preprocessing        clean Price Paid, clean EPC, join on address
02_eda                  what actually correlates with price
03_features             every model feature
04_model                fit and compare versions
src/tune.py             Optuna search (resumable, SQLite-backed)
src/ablation.py         does each feature group earn its place?
src/calibrate.py        prediction intervals + confidence flag
```

## Next steps, in order of value

1. **`05_predict_property.ipynb` — the actual deliverable, and the only
   thing genuinely missing.** Everything it needs exists: model, intervals,
   flag, SHAP. Input postcode + house number (asking price optional, used
   only for the verdict — never as a model input, that would be circular).
2. **Comparable sales** — median price of the *k* nearest **prior** sales,
   type-matched. The evidence points here: missing sale history is the
   single strongest marker of a bad prediction, and sale history covers only
   46% of properties whereas comparables would cover ~100%.
3. **Static EPC condition flags** — room heaters, wall/roof/window ratings,
   SAP gap. Targets cheap properties, which is where the error tail sits.
4. **Floor level for flats** — flats are the worst dwelling type (13.3% vs
   9.1% for houses). Low priority unless buying one.

Judged **not** worth it: motorway distances and city-centre distance (the
whole location group was only worth 0.4pp), time-decay weighting, and
re-tuning (the model converges at 2,034 of 3,000 trees, so capacity isn't
binding).

## Things we learned the hard way

**Write the leakage assertion before the feature.** One assertion —
"no previous sale may be dated on or after its target sale" — caught three
separate real bugs: Land Registry recording one sale twice under two
transaction IDs; the address matcher collapsing several flats in a building
onto one UPRN, so a *neighbour's* sale became this property's history; and
a dedupe that normalised postcodes one line *after* keying on them. **None
would have raised an error.** All three would have made the model look
better than it is by feeding it a near-perfect predictor derived from the
answer. Sale history is now keyed on address identity, not UPRN.

**Correlation is a bad way to choose features.** It misled us three times:
`relative_size` looked weak at 0.25 and cut error 20.1%→18.0%; coastal
distance looked dead at −0.07 and became 13.6% of what the tree uses; floor
area's correlation *fell* when we widened the data and the model got
better. Ablation — fit with and without, measure held-out error — is the
test.

**Check the data covers the same ground as the map.** West Lancashire is in
Lancashire, but the OSM extract was Merseyside — so its distances were
measured against an incomplete map. Dropping it made the extract exactly
correct.

**Verify a query in isolation before committing to a 30-minute run.** Four
consecutive feature runs died on trivial errors (missing import, dtype
mismatch, an OSM layer where `railway` lives inside `other_tags` rather
than being a column). Each cost a full cycle.

**Don't run long jobs inside notebook cells.** An in-notebook Optuna search
hit a 4-hour timeout and lost everything. As a script writing to SQLite it
takes 11 minutes and resumes after interruption.

**Report what didn't work.** Bias correction was built, measured, and
rejected — it made both tails *worse*. The scope experiment reversed twice
before tuning settled it. A results file listing only wins is not much use
later.

## Known limits no amount of tuning fixes

- **Cosmetic condition is invisible** in every free source. We detect
  extensions and efficiency upgrades, not kitchens, bathrooms or damp. This
  is the single biggest missing predictor, worth £20–30k on a £300k house.
- **No bathroom count** exists in UK open data.
- **Lease length** is absent — worth £20–40k on a flat, and a blind spot we
  share with Rightmove's own tool.
- **13% of matched sales have no EPC**, so no floor area. Disproportionately
  long-held owner-occupied homes, missing from both datasets at once.
- **5% error on every property is not achievable on free data.** Zillow
  manages ~7% off-market with licensed bedrooms, bathrooms, floorplans and
  photos. The 5% target is met *conditionally*, inside the green band.

## Manual steps if rebuilding

Two downloads need an account and can't be automated:

- **EPC certificates** — [get-energy-performance-data.communities.gov.uk](https://get-energy-performance-data.communities.gov.uk/),
  free GOV.UK One Login, filter to the councils before downloading
  (~123 MB, not the 7.5 GB national file). Save as
  `data/raw/epc/north_west_certificates.csv`.
- **UBDC Price Paid → UPRN lookup** — [data.ubdc.ac.uk](https://data.ubdc.ac.uk/),
  free account. Save as `data/raw/ubdc/ppdid_uprn_usrn.csv`.

Schools need two files, and note **GIAS does not contain Ofsted ratings** —
they publish separately and join on URN. Ofsted grading is also
mid-transition, so the score has a three-tier fallback (new report card →
legacy grade → Section 8 "remains Good" outcome), which lifts Merseyside
coverage from 56% to 84%. The legacy grade is stored **inverted**
(1 = Outstanding), which is an easy way to get it exactly backwards.
