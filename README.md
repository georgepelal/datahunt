# DataHunt: An ML-Ready Data Acquisition Pipeline (Track: GovAI)

**Course:** ECE 461 — Machine Learning in the Era of Large Language Models  
**Instructor:** Prof. Elias Houstis  
**Student:** Giorgos Pelalidis (AEM: 04136)  
**Institution:** Department of Electrical and Computer Engineering, University of Thessaly  

---

## ⚡ Quick Start: One-Command Reproduction

The entire pipeline (data ingestion, polite caching, provenance sidecar generation, multi-source integration, Parquet export, join-loss auditing, and baseline ML evaluation) reproduces end-to-end with a single command:

```bash
python3 -m datahunt.run --track govai
```

---

## 🧪 Automated Testing Suite

Run the full pytest validation suite (7 passing tests covering caching, JSON-stat decoding, Diavgeia schema validation, join auditing, and temporal evaluation):

```bash
pytest tests/ -v
```

---

## 📁 Repository Structure

```text
datahunt/
├── README.md              # Reproduction guide and architecture overview
├── DATASHEET.md           # Dataset documentation following Gebru et al. (2021)
├── requirements.txt       # Python dependencies
├── sources.csv            # R1: 7-source candidate audit and licence rationale
├── datahunt/
│   ├── fetch.py           # R4: Polite HTTP client with caching & SHA-256 sidecars
│   ├── eurostat_io.py     # R2: JSON-stat 2.0 multidimensional cube decoder
│   ├── other_api.py       # R2: World Bank macroeconomic indicators API
│   ├── unstructured.py    # R3, R6: Diavgeia public decisions ingestion & TF-IDF NLP
│   ├── clean.py           # R5: Cleaning, NUTS-2 integration & join loss audit
│   ├── model.py           # R7: Leakage-free temporal split & Ridge/RF baseline with bootstrap CI
│   └── run.py             # R9: Master CLI reproduction orchestrator
├── data/
│   ├── raw/               # Downloaded payloads with cryptographic sidecars (.json)
│   └── processed/         # ML-ready Parquet tables & audit reports
├── tests/
│   └── test_datahunt.py   # R9: Comprehensive pytest suite
└── report/                # R10: LaTeX report, BibTeX bibliography, figures, compiled PDF
    ├── main.tex
    ├── references.bib
    ├── figures/
    └── main.pdf
```

---

## 🏛️ Track Details (GovAI)

* **Target ML Problem:** Regional economic indicator modeling and unemployment forecasting across Greece's 13 NUTS-2 regions.
* **Structured Sources:**
  - **Eurostat Regional Unemployment** (`lfst_r_lfu3rt`): Annual regional unemployment series.
  - **Eurostat Regional GDP** (`nama_10r_2gdp`): Gross Domestic Product at market prices.
  - **World Bank National Indicators** (`SL.UEM.TOTL.ZS`): National labour market baseline.
* **Unstructured Source:**
  - **Diavgeia Open Governance Decisions** (`https://diavgeia.gov.gr/opendata/search.json`): 500+ ministerial acts, regional public procurements, and expenditure approvals under CC BY 4.0 (PSI Directive).
* **Bonus (+5 Points):**
  - High-resolution NUTS-2 regional choropleth map of Greece using official Eurostat GISCO administrative boundary GeoJSON.
