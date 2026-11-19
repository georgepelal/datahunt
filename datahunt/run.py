"""
datahunt/run.py - Master Pipeline Orchestrator (R9).
Reproduces the complete end-to-end DataHunt pipeline in a single command:
  python -m datahunt.run --track govai
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from datahunt.clean import clean_and_integrate
from datahunt.model import train_and_evaluate_baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DataHunt: ML-ready data acquisition and processing pipeline (ECE 461)."
    )
    parser.add_argument(
        "--track",
        type=str,
        default="govai",
        choices=["govai", "bankai", "healthai", "shopai", "energy"],
        help="Track selection (default: govai)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory storing raw files and provenance sidecar JSONs (default: data/raw)"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory storing processed Parquet tables (default: data/processed)"
    )
    parser.add_argument(
        "--split-year",
        type=int,
        default=2022,
        help="Year threshold for temporal train/test split (default: 2022)"
    )
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print(f"🚀 DataHunt Pipeline Execution — Track: {args.track.upper()}")
    print("=" * 70)

    raw_path = Path(args.raw_dir)
    proc_path = Path(args.processed_dir)

    print(f"\n[1/3] Running multi-source data ingestion, cleaning & integration...")
    print(f"  • Raw directory: {raw_path}")
    print(f"  • Processed directory: {proc_path}")

    df_integrated, join_audit = clean_and_integrate(raw_dir=raw_path, processed_dir=proc_path)
    print(f"  ✅ Integrated dataset produced: {df_integrated.shape[0]} rows x {df_integrated.shape[1]} columns.")
    print(f"  ✅ Parquet saved to: {proc_path / 'integrated_dataset.parquet'}")

    print("\n[2/3] Multi-Source Join-Loss Audit:")
    j1 = join_audit.get("join_1_eurostat_gdp", {})
    print(f"  • Join 1 (Eurostat Unemployment ⨝ GDP): {j1.get('merged_rows')}/{j1.get('left_rows')} rows ({j1.get('pct_retained')}% retained)")
    j2 = join_audit.get("join_2_world_bank", {})
    print(f"  • Join 2 (National Macro Reference ⨝ Regional Panel): {j2.get('merged_rows')}/{j2.get('left_rows')} rows ({j2.get('pct_retained')}% retained)")
    nlp_stats = join_audit.get("governance_nlp_stats", {})
    print(f"  • Unstructured Ingestion (Diavgeia): {nlp_stats.get('total_decisions_sampled')} decisions, €{nlp_stats.get('total_award_amount_eur', 0):,.2f} total expenditure")

    print("\n[3/3] Training & evaluating baseline ML models (R7)...")
    model_results = train_and_evaluate_baseline(
        parquet_path=proc_path / "integrated_dataset.parquet",
        split_year=args.split_year
    )
    ridge_res = model_results["models"]["ridge_regression"]
    rf_res = model_results["models"]["random_forest"]

    print(f"  ✅ Temporal split: Train {model_results['split_config']['train_years']} ({model_results['split_config']['train_observations']} obs) | Test {model_results['split_config']['test_years']} ({model_results['split_config']['test_observations']} obs)")
    print(f"  • Ridge Regression: MAE = {ridge_res['mae']:.2f} (95% CI: [{ridge_res['mae_95_ci'][0]:.2f}, {ridge_res['mae_95_ci'][1]:.2f}]) | RMSE = {ridge_res['rmse']:.2f}")
    print(f"  • Random Forest:    MAE = {rf_res['mae']:.2f} | RMSE = {rf_res['rmse']:.2f}")

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"✨ DataHunt pipeline completed successfully in {elapsed:.2f}s!")
    print("=" * 70)


if __name__ == "__main__":
    main()
