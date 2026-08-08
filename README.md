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
| `03_features.ipynb` | Builds every model feature (inflation adjustment, geocoding, relative size, house type, coastal distance) |
| `04_model.ipynb` | Fits and compares model versions on a temporal train/test split |

## Data sources

All free and open (OGL), no paid APIs:

- **[HM Land Registry Price Paid](https://www.gov.uk/guidance/about-the-price-paid-data)** — every residential sale in England & Wales. Gives the target variable (price).
- **[EPC domestic certificates](https://get-energy-performance-data.communities.gov.uk/)** — floor area, rooms, property type, age band. Requires a free GOV.UK One Login; download filtered to a single council.
- **[UK House Price Index](https://landregistry.data.gov.uk/app/ukhpi)** — monthly index used to restate historic sales in today's money.
- **[UBDC Price Paid → UPRN lookup](https://data.ubdc.ac.uk/)** — published address-matching lookup (free account). Covers 1995–Jan 2022.
- **[postcodes.io](https://postcodes.io/)** — postcode → coordinates, no key needed.
- **[Geofabrik OSM extract](https://download.geofabrik.de/europe/great-britain/england/merseyside.html)** — beaches, Marine Lake, for coastal distance.

## Results so far

Measured on **held-out future sales** (train up to mid-2024, test after) —
so these are honest out-of-sample numbers, not scores on data the model
already saw.

| Model version | Test R² | MdAPE | PPE10 |
|---|---|---|---|
| v0: floor area only, no inflation adjustment | ≈ 0.00 | 33.6% | 10.0% |
| v1: + inflation adjustment | 0.49 | 20.1% | 25.5% |
| v2: + relative size (vs 15 nearest neighbours) | 0.56 | 18.0% | 28.4% |
| v3: + house type + coastal distance | **0.67** | **15.1%** | **34.7%** |

**MdAPE** = median absolute percentage error — typical error size. **PPE10** =
share of predictions within 10% of the true price (the standard automated-
valuation benchmark).

### Findings worth noting

- **The address join reached 95.3%** (vs ~93% in published work). Flats
  matched worst at 75%, and post-2022 sales are weaker because the UBDC
  lookup stops at Jan 2022, leaving the fuzzy matcher to carry those alone.
- **Inflation adjustment mattered more than any feature.** Sefton's median
  price rose 63% over the period; without correcting for that, the model
  scored ≈0 on future sales despite looking fine on training data.
- **EPC energy ratings barely predict price** — but room counts and house
  type predict it strongly. Detached homes sell for roughly £295k against
  £95–125k for terraces.
- **`relative_size` helped, but not the way the hypothesis predicted.** It
  was meant to rescue prediction on uniform-size streets; it actually helped
  most on *varied* streets. Postcode district is probably too coarse a
  neighbourhood definition to test the original claim properly.

## Known limitations

- Sefton only so far — not yet widened to neighbouring authorities.
- No uncertainty estimate yet. The model gives a point prediction, not a
  range, and has no "I can't call this one" refusal rule.
- **14.7% of matched sales have no EPC at all**, so no floor area. These are
  disproportionately long-held owner-occupied homes — the same properties
  missing from both datasets.
- Condition is unobserved. There's no free UK equivalent of a build-quality
  grade, which is the single biggest missing predictor.

## Setup

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
venv/Scripts/python.exe -m ipykernel install --user --name house_price_prediction
```

Then run `00_setup_and_download.ipynb` — it lists the two manual steps
(GOV.UK One Login for EPC, free UBDC account for the UPRN lookup).
