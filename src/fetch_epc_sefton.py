"""
One-off script: pull every domestic EPC certificate in Sefton via the API
and cache to disk. Shares data/raw/epc/ with notebook 02, which reads the
same cache - this script just does the slow part in the background so the
notebook itself stays fast to re-run.

Safe to interrupt and re-run: already-cached certificates are skipped.
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_RAW, EPC_SEARCH_ENDPOINT, EPC_CERTIFICATE_ENDPOINT, EPC_SEARCH_PAGE_SIZE

from dotenv import load_dotenv
import os
import requests
import pandas as pd

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
token = os.environ["EPC_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

REQUESTS_PER_SEC = 16  # 6000/5min = 20/sec limit; leaves headroom for jitter

epc_raw_dir = DATA_RAW / "epc"
epc_raw_dir.mkdir(exist_ok=True, parents=True)
detail_dir = epc_raw_dir / "certificates"
detail_dir.mkdir(exist_ok=True)


def get_search_results() -> pd.DataFrame:
    cache = epc_raw_dir / "sefton_search_results.parquet"
    if cache.exists():
        print(f"[search] loaded cached results from {cache}", flush=True)
        return pd.read_parquet(cache)

    rows = []
    page = 1
    while True:
        resp = requests.get(
            EPC_SEARCH_ENDPOINT,
            headers=HEADERS,
            params={"council[]": "Sefton", "current_page": page, "page_size": EPC_SEARCH_PAGE_SIZE},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        rows.extend(body["data"])
        total_pages = body["pagination"]["totalPages"]
        print(f"[search] page {page}/{total_pages}, running total {len(rows):,}", flush=True)
        if body["pagination"]["nextPage"] is None:
            break
        page += 1
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    print(f"[search] saved {len(df):,} rows to {cache}", flush=True)
    return df


def fetch_all_details(cert_numbers: list[str]) -> None:
    already = sum((detail_dir / f"{c}.json").exists() for c in cert_numbers)
    todo = len(cert_numbers) - already
    print(f"[detail] {already:,} cached, {todo:,} to fetch", flush=True)

    fresh_calls = 0
    t0 = time.time()
    for i, cert_number in enumerate(cert_numbers):
        dest = detail_dir / f"{cert_number}.json"
        if dest.exists():
            continue
        resp = requests.get(
            EPC_CERTIFICATE_ENDPOINT,
            headers=HEADERS,
            params={"certificate_number": cert_number},
            timeout=30,
        )
        if resp.status_code == 404:
            dest.write_text("null", encoding="utf-8")  # cache the miss too, don't retry forever
        else:
            resp.raise_for_status()
            dest.write_text(json.dumps(resp.json()), encoding="utf-8")

        fresh_calls += 1
        target = fresh_calls / REQUESTS_PER_SEC
        elapsed = time.time() - t0
        if elapsed < target:
            time.sleep(target - elapsed)

        if fresh_calls % 500 == 0:
            rate = fresh_calls / elapsed
            remaining = todo - fresh_calls
            eta_min = remaining / rate / 60 if rate > 0 else float("inf")
            print(f"[detail] {i+1:,}/{len(cert_numbers):,} processed, "
                  f"{fresh_calls:,} fresh calls, {rate:.1f}/sec, ETA {eta_min:.0f} min", flush=True)

    print(f"[detail] done. {fresh_calls:,} fresh calls this run.", flush=True)


if __name__ == "__main__":
    search_df = get_search_results()
    fetch_all_details(search_df["certificateNumber"].tolist())
    print("COMPLETE", flush=True)
