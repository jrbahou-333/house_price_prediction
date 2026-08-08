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
# Sefton is the buying target; the rest give one contiguous North West
# market. Measured, not assumed: training on all of Sefton beat training on
# Crosby alone when both were scored on the same Crosby properties (MdAPE
# 9.5% vs 10.0%), so wider training helps rather than dilutes - see the
# scope experiment in notebooks/04_model.ipynb.
#
# Keys are the district name as it appears in Price Paid; values are the
# same authority's RegionName in the UK HPI file. They mostly match but
# must be kept explicit - applying Sefton's price index to Liverpool sales
# would inject error rather than remove it.
LOCAL_AUTHORITIES = {
    "SEFTON": "Sefton",
    "LIVERPOOL": "Liverpool",
    "KNOWSLEY": "Knowsley",
    "WIRRAL": "Wirral",
    "ST HELENS": "St Helens",
    "WEST LANCASHIRE": "West Lancashire",
}

# Slug used for the per-authority EPC download filenames, e.g.
# data/raw/epc/west_lancashire_certificates.csv
def la_slug(district: str) -> str:
    return district.lower().replace(" ", "_")

def epc_path(district: str):
    return DATA_RAW / "epc" / f"{la_slug(district)}_certificates.csv"

def available_authorities() -> list[str]:
    """Whichever authorities actually have an EPC file downloaded.

    Lets the pipeline run on whatever is present rather than failing on a
    missing download - useful while the five extra EPC files are still
    being fetched manually.
    """
    return [d for d in LOCAL_AUTHORITIES if epc_path(d).exists()]

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

# --- EPC ---
# Preferred route is the website's "Download files" flow, which does let you
# filter to a single council before downloading (~120 MB per authority)
# rather than taking the whole-country file (~7.5 GB zipped). Needs a free
# GOV.UK One Login.
#
# The API below is the fallback: same One Login, a bearer token instead of a
# file. Its /api/files/domestic/csv endpoint has no council filter, so the
# only filtered route is search-then-fetch, one call per certificate - a
# couple of hours per authority even paced under the rate limit.
EPC_API_BASE = "https://api.get-energy-performance-data.communities.gov.uk"
EPC_SEARCH_ENDPOINT = f"{EPC_API_BASE}/api/domestic/search"
EPC_CERTIFICATE_ENDPOINT = f"{EPC_API_BASE}/api/certificate"
EPC_SEARCH_PAGE_SIZE = 5000  # API max
EPC_RATE_LIMIT_PER_5MIN = 6000  # per originating IP, across all endpoints
