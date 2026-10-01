"""
datahunt/other_api.py - Secondary structured source ingestion for GovAI (R2).
Queries public governance and macroeconomic data (World Bank / data.gov.gr).
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from datahunt.fetch import fetch

WORLD_BANK_BASE = "https://api.worldbank.org/v2/country/GRC/indicator"
WB_LICENCE = "CC BY 4.0 (World Bank Open Data)"

def get_world_bank_indicator(
    indicator: str = "SL.UEM.TOTL.ZS",
    raw_dir: Path | str = "data/raw"
) -> pd.DataFrame:
    """
    Fetch historical national-level indicator for Greece from World Bank API.
    Indicators:
      - SL.UEM.TOTL.ZS: Unemployment, total (% of total labor force)
      - NY.GDP.MKTP.KD.ZG: GDP growth (annual %)
    """
    url = f"{WORLD_BANK_BASE}/{indicator}"
    params = {"format": "json", "per_page": "100"}
    bin_path = fetch(url, params=params, licence=WB_LICENCE, raw_dir=raw_dir)
    raw_data = json.loads(bin_path.read_text(encoding="utf-8"))

    if len(raw_data) < 2 or not isinstance(raw_data[1], list):
        raise ValueError(f"Unexpected World Bank API payload format for indicator {indicator}")

    records = []
    for item in raw_data[1]:
        val = item.get("value")
        if val is not None:
            records.append({
                "country": item["country"]["id"],
                "year": str(item["date"]),
                "wb_value": float(val),
                "indicator_id": indicator,
                "indicator_name": item["indicator"]["value"]
            })

    df = pd.DataFrame(records)
    return df
