# Datasheet for Dataset: DataHunt-GovAI (Greece NUTS-2 Regional Panel & Public Decisions)

*Following the framework proposed by Gebru et al. (2021), "Datasheets for Datasets", Communications of the ACM.*

---

## 1. Motivation

* **For what purpose was the dataset created?**  
  The dataset was created for academic research in **ECE 461: Machine Learning in the Era of Large Language Models** (University of Thessaly, Fall 2026). Its primary machine learning objective is to study and benchmark predictive models for regional economic indicators (unemployment rate, GDP dynamics) across Greece's 13 NUTS-2 administrative regions by fusing structured official macroeconomic statistics with unstructured public governance decision streams.

* **Who created the dataset?**  
  The dataset pipeline and integrated panel were authored by Giorgos Pelalidis (AEM: 04136) and team as part of Project 1 (DataHunt).

* **Who funded the creation of the dataset?**  
  Academic project conducted within the Department of Electrical and Computer Engineering, University of Thessaly. No commercial or external funding.

---

## 2. Composition

* **What do the instances that comprise the dataset represent?**  
  The integrated processed dataset (`data/processed/integrated_dataset.parquet`) comprises panel observations indexed by **(NUTS-2 Region, Year)** covering 2018–2024 across the 13 Greek administrative regions:
  - `EL30`: Attiki
  - `EL41`: Voreio Aigaio
  - `EL42`: Notio Aigaio
  - `EL43`: Kriti
  - `EL51`: Anatoliki Makedonia, Thraki
  - `EL52`: Kentriki Makedonia
  - `EL53`: Dytiki Makedonia
  - `EL54`: Ipeiros
  - `EL61`: Thessalia
  - `EL62`: Ionia Nisia
  - `EL63`: Dytiki Ellada
  - `EL64`: Sterea Ellada
  - `EL65`: Peloponnisos

* **How many instances are there in total?**  
  - **Structured Panel:** 91 region-year observations across 9 features.
  - **Unstructured Corpus:** 500 public governance acts, procurements, and ministerial decisions collected from Diavgeia, representing over €4.68 million in recorded public expenditure.

* **Does the dataset contain all possible instances or a sample?**  
  - For Eurostat indicators, it contains the complete universe of official regional indicators published by Eurostat for Greece at NUTS-2.
  - For Diavgeia, it contains a verified sample of 500 decisions demonstrating unstructured ingestion, natural language processing, and feature extraction.

* **What data does each instance consist of?**  
  Key attributes include:
  - `geo`: NUTS-2 geographic identifier code.
  - `region_name`: Standardized English transliteration of the Greek region.
  - `year`: Observation year (2018–2024).
  - `unemployment_rate`: Annual regional unemployment rate (%) from Eurostat `lfst_r_lfu3rt`.
  - `gdp_mio_eur`: Regional GDP at market prices (Million EUR) from Eurostat `nama_10r_2gdp`.
  - `log_gdp`: Natural log transform $\ln(1 + \text{GDP})$.
  - `gdp_growth_pct`: Year-over-year regional GDP percentage change.
  - `wb_nat_unemp_rate`: National reference unemployment rate from World Bank.
  - `sample_gov_procurements_mean`: Mean procurement expenditure from Diavgeia sample.

* **Is there any confidential or personally identifiable information (PII)?**  
  No. All data originates from public open-data portals. Personal data was audited in `sources.csv` and marked **No**. Public sector signers and ministers are official authorities acting in their public capacity.

---

## 3. Collection Process

* **What mechanisms or procedures were used to collect the data?**  
  - Automated REST and JSON-stat 2.0 API extraction via custom Python client (`datahunt.fetch`).
  - Strict compliance with web etiquette: Custom `User-Agent` identifying the research affiliation, polite request rate-limiting (minimum 1.0 s sleep delay), exponential backoff on retries.
  - Full provenance caching: Every payload is written to `data/raw/<hash>.bin` accompanied by a sidecar `<hash>.json` recording original URL, parameters, timestamp, HTTP status, byte size, and full SHA-256 cryptographic digest.

* **Who was involved in the data collection process?**  
  Authors using automated reproducible scripts.

* **Over what timeframe was the data collected?**  
  October 2026. The historical statistics span 2018 to 2024.

* **Were any ethical review processes conducted?**  
  All sources are open data published under Creative Commons Attribution 4.0 International (CC BY 4.0) and the European Open Data Directive (EU 2019/1024).

---

## 4. Preprocessing / Cleaning / Labeling

* **Was any preprocessing/cleaning/labeling performed?**  
  Yes:
  1. Multidimensional cube decoding: JSON-stat 2.0 modular arithmetic unpacking.
  2. Geographic and temporal normalization: Alignment of Eurostat NUTS-2 codes and Gregorian calendar years.
  3. Greek text normalization: Stripping of diacritics/tonos (`unicodedata.normalize`), punctuation removal, lowercase conversion, and Greek stop-word filtering.
  4. Missing values and flag handling: Proper handling of Eurostat provisional flags; temporal forward and backward imputation for regional differencing.
  5. Multi-source join auditing: Documented join loss showing 87.5% retention on GDP (due to official publication lag of the most recent year) and 100% on national indicators.

---

## 5. Uses

* **Has the dataset been used for any tasks already?**  
  Used to train and benchmark regularized Ridge Regression and Random Forest models for regional unemployment forecasting with temporal out-of-time evaluation (pre-2022 train, post-2022 test) and bootstrap confidence intervals.

* **What other tasks could the dataset be used for?**  
  Macroeconomic policy evaluation, regional divergence tracking, text-based public spending sentiment and topic modeling.

* **What tasks should the dataset NOT be used for?**  
  It should NOT be used for high-stakes individual employment decisions, credit scoring, or commercial financial speculation.

---

## 6. Distribution

* **How will the dataset be distributed?**  
  Open academic repository with full code pipeline (`python -m datahunt.run --track govai`) and Parquet table in `data/processed/integrated_dataset.parquet`.

* **Under what licence is the dataset distributed?**  
  Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 7. Maintenance

* **Who is maintaining the dataset?**  
  Giorgos Pelalidis (contact: `gpelalidis@uth.gr`), University of Thessaly.

* **How can the owner/curator be contacted?**  
  Via University of Thessaly institutional email.
