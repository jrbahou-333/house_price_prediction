"""
Shared paths and constants for the Crosby house price project.
See the plan: notebooks/00_setup_and_download.ipynb explains what goes where.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

for _d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)

# --- Local authorities in scope (Part 3 of the plan) ---
# Sefton is the buying target; the rest give one contiguous North West market
# with enough transaction volume for gradient boosting.
LOCAL_AUTHORITIES = [
    "SEFTON",
    "LIVERPOOL",
    "KNOWSLEY",
    "WIRRAL",
    "ST HELENS",
    "WEST LANCASHIRE",
]

# Phase 1 thin slice uses Sefton only.
PHASE1_LA = ["SEFTON"]

# EPC certificates don't exist before 2008, so there is no point pulling
# Price Paid data further back than that for the join.
EPC_START_YEAR = 2008

# HM Land Registry Price Paid Data has no header row. This is the fixed
# column order (see https://www.gov.uk/guidance/about-the-price-paid-data).
PPD_COLUMNS = [
    "transaction_id",
    "price",
    "date_of_transfer",
    "postcode",
    "property_type",  # D/S/T/F/O
    "old_new",  # Y/N
    "duration",  # F/L (freehold/leasehold)
    "paon",
    "saon",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category_type",  # A = standard price paid, B = additional (drop these)
    "record_status",  # A/C/D - monthly files only
]

PPD_YEARLY_URL = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-{year}.csv"
PPD_COMPLETE_URL = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-complete.csv"

# --- EPC: API, not bulk CSV download ---
# The bulk CSV download is now whole-country only (~8GB) since per-LA
# download was removed from the service. The API lets us filter server-side
# by council instead - same GOV.UK One Login requirement for the bearer
# token, but no multi-GB file. Two-step: search (lightweight, paginated,
# gives certificate numbers) then fetch full detail per certificate.
EPC_API_BASE = "https://api.get-energy-performance-data.communities.gov.uk"
EPC_SEARCH_ENDPOINT = f"{EPC_API_BASE}/api/domestic/search"
EPC_CERTIFICATE_ENDPOINT = f"{EPC_API_BASE}/api/certificate"
EPC_SEARCH_PAGE_SIZE = 5000  # API max
EPC_RATE_LIMIT_PER_5MIN = 6000  # per originating IP, across all endpoints
