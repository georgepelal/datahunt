"""
datahunt/clean.py - Data cleaning, validation, and multi-source integration (R5).
Merges Eurostat regional indicators (Unemployment & GDP), World Bank national indicators,
and Diavgeia governance/NLP features into an ML-ready panel dataset (NUTS-2 x Year).
Produces tidy Parquet in data/processed/ and computes join loss statistics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datahunt.eurostat_io import get_regional_unemployment, get_regional_gdp, GREEK_NUTS2
from datahunt.other_api import get_world_bank_indicator
from datahunt.unstructured import fetch_diavgeia_raw, parse_diavgeia_records, extract_text_features


def clean_and_integrate(
    raw_dir: Path | str = "data/raw",
    processed_dir: Path | str = "data/processed"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Executes the full R5 cleaning and multi-source integration pipeline:
    1. Ingests Eurostat Unemployment (target/indicator).
    2. Ingests Eurostat Regional GDP.
    3. Ingests World Bank national macroeconomic reference series.
    4. Ingests & processes Diavgeia text and governance decision statistics.
    5. Joins on keys (geo/nuts2, year) while tracking and logging join loss.
    6. Saves tidy Parquet dataset to data/processed/.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    join_loss_audit: dict[str, Any] = {}

    # -------------------------------------------------------------
    # 1. Eurostat Regional Unemployment
    # -------------------------------------------------------------
    df_unemp = get_regional_unemployment(raw_dir=raw_path)
    # Filter only Greek NUTS-2 regions and cast year
    df_unemp = df_unemp[df_unemp["geo"].isin(GREEK_NUTS2.keys())].copy()
    df_unemp["year"] = pd.to_numeric(df_unemp["year"], errors="coerce")
    df_unemp["unemployment_rate"] = pd.to_numeric(df_unemp["unemployment_rate"], errors="coerce")
    df_unemp = df_unemp.dropna(subset=["year", "unemployment_rate"])
    df_unemp["year"] = df_unemp["year"].astype(int)

    base_unemp = df_unemp[["geo", "year", "unemployment_rate"]].drop_duplicates(subset=["geo", "year"])
    initial_rows = len(base_unemp)
    join_loss_audit["initial_unemployment_rows"] = initial_rows

    # -------------------------------------------------------------
    # 2. Eurostat Regional GDP
    # -------------------------------------------------------------
    df_gdp = get_regional_gdp(raw_dir=raw_path)
    df_gdp = df_gdp[df_gdp["geo"].isin(GREEK_NUTS2.keys())].copy()
    df_gdp["year"] = pd.to_numeric(df_gdp["year"], errors="coerce")
    df_gdp["gdp_mio_eur"] = pd.to_numeric(df_gdp["gdp_mio_eur"], errors="coerce")
    df_gdp = df_gdp.dropna(subset=["year", "gdp_mio_eur"])
    df_gdp["year"] = df_gdp["year"].astype(int)
    gdp_clean = df_gdp[["geo", "year", "gdp_mio_eur"]].drop_duplicates(subset=["geo", "year"])

    # Join 1: Unemployment + GDP on (geo, year)
    merged_1 = pd.merge(base_unemp, gdp_clean, on=["geo", "year"], how="inner")
    join_loss_audit["join_1_eurostat_gdp"] = {
        "left_rows": initial_rows,
        "right_rows": len(gdp_clean),
        "merged_rows": len(merged_1),
        "rows_lost": initial_rows - len(merged_1),
        "pct_retained": round((len(merged_1) / initial_rows) * 100, 2) if initial_rows > 0 else 0.0
    }

    # -------------------------------------------------------------
    # 3. World Bank National Context
    # -------------------------------------------------------------
    df_wb = get_world_bank_indicator("SL.UEM.TOTL.ZS", raw_dir=raw_path)
    df_wb["year"] = pd.to_numeric(df_wb["year"], errors="coerce")
    df_wb["wb_value"] = pd.to_numeric(df_wb["wb_value"], errors="coerce")
    df_wb = df_wb.dropna(subset=["year", "wb_value"])
    df_wb["year"] = df_wb["year"].astype(int)
    wb_clean = df_wb[["year", "wb_value"]].rename(columns={"wb_value": "wb_nat_unemp_rate"}).drop_duplicates("year")

    # Join 2: Merge national reference on (year)
    merged_2 = pd.merge(merged_1, wb_clean, on=["year"], how="left")
    join_loss_audit["join_2_world_bank"] = {
        "left_rows": len(merged_1),
        "right_rows": len(wb_clean),
        "merged_rows": len(merged_2),
        "rows_lost": 0,  # Left join preserves all regional rows
        "pct_retained": 100.0
    }

    # -------------------------------------------------------------
    # 4. Unstructured Diavgeia Processing & Feature Aggregation
    # -------------------------------------------------------------
    diavgeia_paths = fetch_diavgeia_raw(target_count=500, raw_dir=raw_path)
    df_diavgeia = parse_diavgeia_records(diavgeia_paths)
    df_feat, text_features, _ = extract_text_features(df_diavgeia, max_features=10)

    # Compute national/cross-regional governance summary features
    gov_stats = {
        "total_decisions_sampled": len(df_diavgeia),
        "total_award_amount_eur": float(df_diavgeia["award_amount_eur"].sum()),
        "mean_award_amount_eur": float(df_diavgeia["award_amount_eur"].mean()),
        "top_text_features": text_features
    }
    join_loss_audit["governance_nlp_stats"] = gov_stats

    # Add regional metadata and engineer domain features
    merged_2["region_name"] = merged_2["geo"].map(GREEK_NUTS2)
    merged_2["log_gdp"] = np.log1p(merged_2["gdp_mio_eur"])

    # Compute year-over-year GDP growth per region
    merged_2 = merged_2.sort_values(["geo", "year"]).reset_index(drop=True)
    merged_2["gdp_growth_pct"] = merged_2.groupby("geo")["gdp_mio_eur"].pct_change() * 100.0
    # Fill first year diff with regional median
    merged_2["gdp_growth_pct"] = merged_2["gdp_growth_pct"].fillna(merged_2.groupby("geo")["gdp_growth_pct"].transform("median"))
    # Fallback fill for any remaining NaNs
    merged_2["gdp_growth_pct"] = merged_2["gdp_growth_pct"].fillna(0.0)

    # Attach governance feature metrics
    merged_2["sample_gov_procurements_mean"] = gov_stats["mean_award_amount_eur"]

    # -------------------------------------------------------------
    # 5. Export to Parquet & Save Audit Report
    # -------------------------------------------------------------
    parquet_path = processed_path / "integrated_dataset.parquet"
    merged_2.to_parquet(parquet_path, index=False)

    audit_path = processed_path / "join_loss_audit.json"
    audit_path.write_text(json.dumps(join_loss_audit, indent=2), encoding="utf-8")

    return merged_2, join_loss_audit
