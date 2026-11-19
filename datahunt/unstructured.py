"""
datahunt/unstructured.py - Unstructured data collection and feature extraction (R3 & R6).
Fetches public governance decisions from Diavgeia Open Data API, records provenance,
and extracts NLP features and procurement statistics.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from datahunt.fetch import fetch

DIAVGEIA_SEARCH_URL = "https://diavgeia.gov.gr/opendata/search.json"
DIAVGEIA_LICENCE = "CC BY 4.0 (Diavgeia Open Governance / PSI Directive 2019/1024)"

# Common Greek stop words for administrative texts
GREEK_STOP_WORDS = {
    "και", "του", "της", "των", "την", "τον", "το", "τα", "για", "στο", "στη", "στην",
    "στον", "στους", "στις", "στα", "απο", "με", "κατα", "προς", "περι", "υπερ", "μετα",
    "δια", "επι", "σε", "ως", "ειναι", "ηταν", "θα", "να", "που", "πως", "οτι", "αν",
    "μην", "δεν", "αυτο", "αυτη", "αυτος", "αυτοι", "αυτες", "αυτα", "ο", "η", "οι",
    "εγκριση", "αποφαση", "πραξη", "αριθμ", "πρωτ", "αρ", "δ", "τουτου", "της", "των"
}

def clean_greek_text(text: str) -> str:
    """
    Normalizes Greek text by removing accents/diacritics, punctuation,
    converting to lowercase, and collapsing whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Decompose unicode to remove tonos/dialytika
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    lowered = stripped.lower()
    cleaned = re.sub(r"[^a-zα-ω0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()

def fetch_diavgeia_raw(
    target_count: int = 500,
    page_size: int = 100,
    raw_dir: Path | str = "data/raw",
    sleep: float = 1.0
) -> list[Path]:
    """
    Politely fetches target_count Diavgeia decisions in pages of page_size.
    Every page is saved to raw_dir via datahunt.fetch with full provenance sidecar.
    Returns list of Path objects for all cached page files.
    """
    num_pages = (target_count + page_size - 1) // page_size
    page_paths: list[Path] = []

    for page_idx in range(num_pages):
        params = {
            "size": page_size,
            "page": page_idx
        }
        bin_path = fetch(
            DIAVGEIA_SEARCH_URL,
            params=params,
            licence=DIAVGEIA_LICENCE,
            raw_dir=raw_dir,
            sleep=sleep
        )
        page_paths.append(bin_path)

    return page_paths

def parse_diavgeia_records(raw_page_paths: list[Path]) -> pd.DataFrame:
    """
    Parses downloaded Diavgeia JSON payloads into a structured DataFrame.
    Validates per-item metadata including ADA, subject, timestamp, URL, and licence.
    """
    records: list[dict[str, Any]] = []

    for path in raw_page_paths:
        content = json.loads(path.read_text(encoding="utf-8"))
        decisions = content.get("decisions", [])
        for dec in decisions:
            ada = dec.get("ada")
            subject = dec.get("subject", "")
            ts = dec.get("submissionTimestamp") or dec.get("issueDate")

            # Extract year from unix epoch timestamp (milliseconds)
            if ts:
                dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
                year = dt.year
                date_str = dt.strftime("%Y-%m-%d")
            else:
                year = None
                date_str = None

            # Financial extraction if available in extraFieldValues
            extra = dec.get("extraFieldValues") or {}
            award = extra.get("awardAmount") or {}
            amount = award.get("amount") if isinstance(award, dict) else None
            assignment_type = extra.get("assignmentType") if isinstance(extra, dict) else None

            records.append({
                "ada": ada,
                "subject": subject,
                "cleaned_subject": clean_greek_text(subject),
                "year": year,
                "date": date_str,
                "organization_id": dec.get("organizationId"),
                "decision_type_id": dec.get("decisionTypeId"),
                "assignment_type": assignment_type,
                "award_amount_eur": float(amount) if amount is not None else 0.0,
                "url": dec.get("url"),
                "document_url": dec.get("documentUrl"),
                "licence": DIAVGEIA_LICENCE
            })

    df = pd.DataFrame(records)
    # Deduplicate by ADA
    df = df.drop_duplicates(subset=["ada"]).reset_index(drop=True)
    return df

def extract_text_features(
    df: pd.DataFrame,
    max_features: int = 15,
    min_df: int = 2
) -> tuple[pd.DataFrame, list[str], TfidfVectorizer]:
    """
    Extracts TF-IDF features from decision subjects (R6).
    Appends TF-IDF feature columns to df and returns (df_with_features, feature_names, vectorizer).
    """
    corpus = df["cleaned_subject"].fillna("").tolist()

    # Custom stop words list
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words=list(GREEK_STOP_WORDS),
        token_pattern=r"(?u)\b\w{3,}\b"
    )

    tfidf_matrix = vectorizer.fit_transform(corpus).toarray()
    feature_names = [f"tfidf_{w}" for w in vectorizer.get_feature_names_out()]

    tfidf_df = pd.DataFrame(tfidf_matrix, columns=feature_names, index=df.index)
    result_df = pd.concat([df, tfidf_df], axis=1)

    return result_df, feature_names, vectorizer
