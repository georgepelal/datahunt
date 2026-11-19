"""
tests/test_datahunt.py - Comprehensive pytest test suite for DataHunt (GovAI track).
Validates caching, JSON-stat decoding, unstructured Diavgeia ingestion,
cleaning, join loss audits, and baseline evaluation.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datahunt.fetch import fetch
from datahunt.eurostat_io import jsonstat_to_df, GREEK_NUTS2
from datahunt.unstructured import clean_greek_text, parse_diavgeia_records, extract_text_features
from datahunt.clean import clean_and_integrate
from datahunt.model import train_and_evaluate_baseline


def test_fetch_caching_and_provenance(tmp_path: Path):
    """
    Test 1: Verify polite fetcher caching, payload persistence, and sidecar metadata.
    """
    sample_url = "https://httpbin.org/get"
    licence_tag = "CC BY 4.0 Test"
    raw_dir = tmp_path / "raw"

    # First fetch
    bin_path = fetch(sample_url, licence=licence_tag, raw_dir=raw_dir, sleep=0.0)
    assert bin_path.exists()

    sidecar = bin_path.with_suffix(".json")
    assert sidecar.exists()

    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["licence"] == licence_tag
    assert meta["status"] == 200
    assert "sha256" in meta
    assert len(meta["sha256"]) == 64

    # Second fetch must return cached copy immediately
    bin_path_2 = fetch(sample_url, licence=licence_tag, raw_dir=raw_dir, sleep=0.0)
    assert bin_path == bin_path_2


def test_jsonstat_decoder_logic():
    """
    Test 2: Verify JSON-stat 2.0 multidimensional coordinate decoding and row-major strides.
    """
    synthetic_jsonstat = {
        "version": "2.0",
        "class": "dataset",
        "id": ["geo", "time"],
        "size": [2, 3],
        "dimension": {
            "geo": {
                "category": {
                    "index": ["EL30", "EL41"],
                    "label": {"EL30": "Attiki", "EL41": "Voreio Aigaio"}
                }
            },
            "time": {
                "category": {
                    "index": ["2020", "2021", "2022"],
                    "label": {"2020": "2020", "2021": "2021", "2022": "2022"}
                }
            }
        },
        "value": [10.5, 11.0, 12.0, 15.1, 14.8, 13.9]
    }

    df = jsonstat_to_df(synthetic_jsonstat)
    assert df.shape == (6, 3)
    assert list(df.columns) == ["geo", "time", "value"]

    # Verify first and last decoded rows
    assert df.iloc[0]["geo"] == "EL30"
    assert df.iloc[0]["time"] == "2020"
    assert df.iloc[0]["value"] == 10.5

    assert df.iloc[5]["geo"] == "EL41"
    assert df.iloc[5]["time"] == "2022"
    assert df.iloc[5]["value"] == 13.9


def test_greek_text_cleaning_and_diacritics():
    """
    Test 3: Verify Greek text normalization and diacritic removal for NLP.
    """
    raw_greek = "Έγκριση Προμήθειας Ορθοπεδικών Υλικών & Δαπανών!"
    cleaned = clean_greek_text(raw_greek)
    assert "εγκριση" in cleaned
    assert "προμηθειας" in cleaned
    assert "ορθοπεδικων" in cleaned
    assert "!" not in cleaned
    assert "&" not in cleaned


def test_diavgeia_unstructured_parser(tmp_path: Path):
    """
    Test 4: Verify parsing of Diavgeia JSON payloads and per-item metadata.
    """
    fake_json = {
        "decisions": [
            {
                "ada": "Ψ123456-ABC",
                "subject": "ΑΠΟΦΑΣΗ ΕΓΚΡΙΣΗΣ ΔΑΠΑΝΗΣ ΠΕΡΙΦΕΡΕΙΑΣ ΘΕΣΣΑΛΙΑΣ",
                "submissionTimestamp": 1640995200000,
                "organizationId": "9920",
                "decisionTypeId": "2.4.6.1",
                "url": "https://diavgeia.gov.gr/api/decisions/Ψ123456-ABC",
                "documentUrl": "https://diavgeia.gov.gr/doc/Ψ123456-ABC",
                "extraFieldValues": {
                    "awardAmount": {"amount": 15000.0, "currency": "EUR"},
                    "assignmentType": "Έργα"
                }
            }
        ]
    }
    sample_file = tmp_path / "sample_diavgeia.bin"
    sample_file.write_text(json.dumps(fake_json), encoding="utf-8")

    df = parse_diavgeia_records([sample_file])
    assert len(df) == 1
    assert df.iloc[0]["ada"] == "Ψ123456-ABC"
    assert df.iloc[0]["year"] == 2022
    assert df.iloc[0]["award_amount_eur"] == 15000.0
    assert "CC BY 4.0" in df.iloc[0]["licence"]


def test_nlp_feature_extraction():
    """
    Test 5: Verify TF-IDF feature extraction on cleaned decision subjects.
    """
    df = pd.DataFrame({
        "cleaned_subject": [
            "προμηθεια υπολογιστων και εξοπλισμου πληροφορικης",
            "εργασιες συντηρησης σχολικων κτιριων",
            "προμηθεια αναλωσιμων υλικων γραφειου"
        ]
    })
    df_feat, feat_names, _ = extract_text_features(df, max_features=5, min_df=1)
    assert len(feat_names) > 0
    assert any(col.startswith("tfidf_") for col in df_feat.columns)
    assert not df_feat[feat_names].isna().any().any()


def test_clean_and_integration_pipeline(tmp_path: Path):
    """
    Test 6: Verify end-to-end multi-source join, Parquet serialization, and audit report.
    """
    base_raw = Path("data/raw")
    proc_test_dir = tmp_path / "processed"

    df_integrated, audit = clean_and_integrate(raw_dir=base_raw, processed_dir=proc_test_dir)
    assert not df_integrated.empty
    assert "geo" in df_integrated.columns
    assert "year" in df_integrated.columns
    assert "unemployment_rate" in df_integrated.columns
    assert "gdp_mio_eur" in df_integrated.columns

    # Verify parquet file was saved
    parquet_file = proc_test_dir / "integrated_dataset.parquet"
    assert parquet_file.exists()
    reloaded_df = pd.read_parquet(parquet_file)
    assert len(reloaded_df) == len(df_integrated)

    # Verify audit contains join loss statistics
    assert "join_1_eurostat_gdp" in audit
    assert "pct_retained" in audit["join_1_eurostat_gdp"]


def test_baseline_model_evaluation(tmp_path: Path):
    """
    Test 7: Verify leakage-free temporal split and bootstrap CI evaluation.
    """
    base_raw = Path("data/raw")
    proc_test_dir = tmp_path / "processed"
    clean_and_integrate(raw_dir=base_raw, processed_dir=proc_test_dir)

    results = train_and_evaluate_baseline(
        parquet_path=proc_test_dir / "integrated_dataset.parquet",
        split_year=2022,
        n_bootstraps=100
    )

    # Verify temporal split is strictly segregated
    train_years = results["split_config"]["train_years"]
    test_years = results["split_config"]["test_years"]
    assert max(train_years) < min(test_years)

    # Verify performance metrics & confidence intervals
    ridge = results["models"]["ridge_regression"]
    assert ridge["mae"] > 0
    assert ridge["rmse"] > 0
    assert len(ridge["mae_95_ci"]) == 2
    assert ridge["mae_95_ci"][0] <= ridge["mae_95_ci"][1]
