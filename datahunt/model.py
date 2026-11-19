"""
datahunt/model.py - Baseline Machine Learning Model & Evaluation (R7).
Implements a leakage-free temporal train/test split on the integrated GovAI panel dataset,
trains a regularized linear baseline (Ridge Regression) and Random Forest to predict regional unemployment,
and reports performance metrics (MAE, RMSE, R2) with 95% bootstrap confidence intervals.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def train_and_evaluate_baseline(
    parquet_path: Path | str = "data/processed/integrated_dataset.parquet",
    split_year: int = 2022,
    n_bootstraps: int = 1000,
    random_state: int = 42
) -> dict[str, Any]:
    """
    Trains baseline models using a temporal split:
      - Train set: year < split_year (historical training)
      - Test set: year >= split_year (out-of-time evaluation, zero future leakage)
    Computes MAE, RMSE, R2, and 95% bootstrap confidence intervals for the test set.
    """
    df = pd.read_parquet(parquet_path)

    # Feature columns: macroeconomic variables + one-hot encoded regions
    feature_cols = ["log_gdp", "gdp_growth_pct"]
    
    # One-hot encode regional fixed effects
    df_encoded = pd.get_dummies(df, columns=["geo"], drop_first=True)
    geo_dummy_cols = [c for c in df_encoded.columns if c.startswith("geo_")]
    all_features = feature_cols + geo_dummy_cols

    target_col = "unemployment_rate"

    # Leakage-free temporal split
    train_mask = df_encoded["year"] < split_year
    test_mask = df_encoded["year"] >= split_year

    X_train = df_encoded.loc[train_mask, all_features]
    y_train = df_encoded.loc[train_mask, target_col]
    X_test = df_encoded.loc[test_mask, all_features]
    y_test = df_encoded.loc[test_mask, target_col]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"Invalid temporal split at year {split_year}: train={len(X_train)}, test={len(X_test)}")

    # Standardize continuous features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])

    # Model 1: Ridge Regression
    ridge = Ridge(alpha=1.0, random_state=random_state)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)

    # Model 2: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Evaluation metrics
    mae_ridge = float(mean_absolute_error(y_test, y_pred_ridge))
    rmse_ridge = float(np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
    r2_ridge = float(r2_score(y_test, y_pred_ridge))

    mae_rf = float(mean_absolute_error(y_test, y_pred_rf))
    rmse_rf = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    r2_rf = float(r2_score(y_test, y_pred_rf))

    # Bootstrap 95% Confidence Intervals for Ridge MAE and RMSE
    rng = np.random.default_rng(random_state)
    boot_maes: list[float] = []
    boot_rmses: list[float] = []
    n_test = len(y_test)
    y_test_arr = y_test.to_numpy()

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n_test, size=n_test)
        b_true = y_test_arr[boot_idx]
        b_pred = y_pred_ridge[boot_idx]
        boot_maes.append(mean_absolute_error(b_true, b_pred))
        boot_rmses.append(np.sqrt(mean_squared_error(b_true, b_pred)))

    ci_mae_low, ci_mae_high = np.percentile(boot_maes, [2.5, 97.5])
    ci_rmse_low, ci_rmse_high = np.percentile(boot_rmses, [2.5, 97.5])

    results = {
        "split_config": {
            "split_year": split_year,
            "train_observations": len(X_train),
            "test_observations": len(X_test),
            "train_years": sorted(df.loc[train_mask, "year"].unique().tolist()),
            "test_years": sorted(df.loc[test_mask, "year"].unique().tolist())
        },
        "models": {
            "ridge_regression": {
                "mae": round(mae_ridge, 4),
                "rmse": round(rmse_ridge, 4),
                "r2": round(r2_ridge, 4),
                "mae_95_ci": [round(float(ci_mae_low), 4), round(float(ci_mae_high), 4)],
                "rmse_95_ci": [round(float(ci_rmse_low), 4), round(float(ci_rmse_high), 4)]
            },
            "random_forest": {
                "mae": round(mae_rf, 4),
                "rmse": round(rmse_rf, 4),
                "r2": round(r2_rf, 4)
            }
        }
    }

    # Save evaluation summary
    out_path = Path(parquet_path).parent / "baseline_evaluation.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results
