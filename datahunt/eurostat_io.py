"""
datahunt/eurostat_io.py - Eurostat ingestion via multiple routes & cube decoding (R2).
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from datahunt.fetch import fetch

EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
EUROSTAT_LICENCE = "CC BY 4.0 (Eurostat Open Data)"

# Greek NUTS-2 regions mapping
GREEK_NUTS2 = {
    "EL30": "Attiki",
    "EL41": "Voreio Aigaio",
    "EL42": "Notio Aigaio",
    "EL43": "Kriti",
    "EL51": "Anatoliki Makedonia, Thraki",
    "EL52": "Kentriki Makedonia",
    "EL53": "Dytiki Makedonia",
    "EL54": "Ipeiros",
    "EL61": "Thessalia",
    "EL62": "Ionia Nisia",
    "EL63": "Dytiki Ellada",
    "EL64": "Sterea Ellada",
    "EL65": "Peloponnisos"
}

def jsonstat_to_df(js: dict) -> pd.DataFrame:
    """
    Flatten a JSON-stat 2.0 dataset into a tidy (long) DataFrame.
    Peels off the fastest-varying dimension first using modular arithmetic.
    """
    ids, sizes = js["id"], js["size"]
    codes = []
    for d in ids:
        cat = js["dimension"][d]["category"]
        idx = cat.get("index", [])
        if isinstance(idx, list):
            codes.append(idx)
        elif isinstance(idx, dict):
            codes.append(sorted(idx, key=idx.get))
        else:
            # Fallback if categories are listed as keys
            codes.append(list(cat.get("label", {}).keys()))

    values = js.get("value", {})
    items = values.items() if isinstance(values, dict) else enumerate(values)
    
    rows = []
    for k, v in items:
        k = int(k)
        coord = []
        for n in reversed(sizes):
            coord.append(k % n)
            k //= n
        coord.reverse()
        rows.append([codes[j][i] for j, i in enumerate(coord)] + [v])

    cols = ids + ["value"]
    df = pd.DataFrame(rows, columns=cols)
    return df

def get_eurostat_jsonstat(
    dataset_code: str,
    params: dict | None = None,
    raw_dir: Path | str = "data/raw"
) -> pd.DataFrame:
    """
    Route 1: Query Eurostat Dissemination API using JSON-stat 2.0 with cached fetch.
    """
    url = f"{EUROSTAT_BASE}/{dataset_code}"
    bin_path = fetch(url, params=params, licence=EUROSTAT_LICENCE, raw_dir=raw_dir)
    raw_json = json.loads(bin_path.read_text(encoding="utf-8"))
    df = jsonstat_to_df(raw_json)
    return df

def get_eurostat_tsv(
    dataset_code: str,
    raw_dir: Path | str = "data/raw"
) -> pd.DataFrame:
    """
    Route 2: Direct bulk TSV dissemination route.
    Used for automated route equivalence verification (R2).
    """
    # Alternatively use eurostat python package or direct bulk endpoint
    tsv_url = f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{dataset_code}?format=TSV"
    bin_path = fetch(tsv_url, licence=EUROSTAT_LICENCE, raw_dir=raw_dir)
    return bin_path

def get_regional_unemployment(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """
    Fetch regional unemployment at NUTS-2 level for Greece (dataset: lfst_r_lfu3rt).
    """
    params = {
        "geo": list(GREEK_NUTS2.keys()),
        "unit": "PC",
        "sex": "T",
        "age": "Y15-74",
        "isced11": "TOTAL",
        "sinceTimePeriod": 2018
    }
    df = get_eurostat_jsonstat("lfst_r_lfu3rt", params=params, raw_dir=raw_dir)
    df = df.rename(columns={"time": "year", "value": "unemployment_rate"})
    return df

def get_regional_gdp(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """
    Fetch regional GDP at current market prices for NUTS-2 regions (dataset: nama_10r_2gdp).
    """
    params = {
        "geo": list(GREEK_NUTS2.keys()),
        "unit": "MIO_EUR",
        "sinceTimePeriod": 2018
    }
    df = get_eurostat_jsonstat("nama_10r_2gdp", params=params, raw_dir=raw_dir)
    df = df.rename(columns={"time": "year", "value": "gdp_mio_eur"})
    return df
    df = df.rename(columns={"time": "year", "value": "gdp_mio_eur"})
    return df
