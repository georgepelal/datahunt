"""
datahunt/fetch.py - Polite, cached HTTP client with provenance recording (R4).
"""
from __future__ import annotations

import time
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timezone

HEADERS = {
    "User-Agent": "DataHunt-GovAI/1.0 (Academic Research; ECE461 Univ of Thessaly; contact: gpelalidis@uth.gr)",
    "Accept": "application/json, text/plain, */*"
}

def fetch(
    url: str,
    params: dict | None = None,
    *,
    licence: str,
    raw_dir: Path | str = "data/raw",
    sleep: float = 1.0,
    max_retries: int = 4,
    timeout: int = 60
) -> Path:
    """
    Download url (+params) once into raw_dir, writing a provenance sidecar JSON.
    Returns the Path of the raw binary file. Re-running returns the cached copy.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    # Deterministic SHA-256 cache key based on URL and sorted parameters
    param_str = json.dumps(params or {}, sort_keys=True)
    cache_key = hashlib.sha256((url + param_str).encode("utf-8")).hexdigest()[:16]

    bin_path = raw_path / f"{cache_key}.bin"
    meta_path = raw_path / f"{cache_key}.json"

    # Return cached file if both payload and provenance sidecar exist
    if bin_path.exists() and meta_path.exists():
        return bin_path

    # Polite exponential backoff retry loop
    response = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                backoff = sleep * (2 ** (attempt - 1))
                time.sleep(backoff)
                continue
            r.raise_for_status()
            response = r
            break
        except requests.RequestException as e:
            if attempt == max_retries:
                raise RuntimeError(f"DataHunt fetch failed for {url} after {max_retries} attempts: {e}")
            time.sleep(sleep * (2 ** (attempt - 1)))

    if response is None:
        raise RuntimeError(f"Failed to fetch {url}")

    content = response.content
    bin_path.write_bytes(content)

    meta = {
        "url": response.url,
        "requested_url": url,
        "params": params,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "status": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "licence": licence,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Polite rate limiting between fresh requests
    time.sleep(sleep)
    return bin_path
