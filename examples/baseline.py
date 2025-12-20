#!/usr/bin/env python
"""Benchmark evaluation helpers for launder-simulation exports.

This script demonstrates:
- Loading exported CSVs (accounts, entities, edges, graph features).
- Computing summary statistics such as class imbalance, pattern prevalence,
  top-degree accounts, and risk score distributions.
- Performing a temporal train/validation/test split by day index.
- Training a simple baseline classifier (LightGBM if available, otherwise
  GradientBoostingClassifier) to detect illicit transactions.
- Bootstrapping quick LI/HI simulations to illustrate scenario selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

from aml_sim.config import load_config
from aml_sim.detection import RiskModel
from aml_sim.simulation import Simulation

try:
    import lightgbm as lgb

    HAVE_LIGHTGBM = True
except Exception:
    HAVE_LIGHTGBM = False


def load_exports(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load exported CSVs into pandas DataFrames."""

    paths = {
        "accounts": data_dir / "accounts.csv",
        "entities": data_dir / "entities.csv",
        "edges": data_dir / "graph_edges.csv",
        "features": data_dir / "graph_features.csv",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing exports: {', '.join(missing)} in {data_dir}")
    return {name: pd.read_csv(path) for name, path in paths.items()}


def attach_risk_scores(edges: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Compute risk scores for each transaction using the built-in RiskModel."""

    model = RiskModel(threshold, {})
    scored = edges.copy()
    scored["risk_score"] = scored.apply(
        lambda row: model.score_transaction(
            row["amount"],
            bool(row.get("is_money_laundering", False)),
            row.get("illicit_amount", 0.0),
            row.get("illicit_fraction", 0.0),
            bool(row.get("cross_bank", False)),
            bool(row.get("cross_currency", False)),
            channel=row.get("channel"),
            tx_type=row.get("tx_type"),
        ),
        axis=1,
    )
    return scored


def compute_summary_stats(edges: pd.DataFrame, graph_features: pd.DataFrame) -> Dict[str, object]:
    """Return summary statistics useful for benchmarking."""

    stats: Dict[str, object] = {}
    stats["class_balance"] = edges["is_money_laundering"].value_counts(normalize=True).to_dict()
    stats["pattern_prevalence"] = edges["ml_pattern"].value_counts().to_dict()
    gf = graph_features.copy()
    gf["total_degree"] = gf["in_degree"] + gf["out_degree"]
    stats["top_degree_accounts"] = gf.sort_values("total_degree", ascending=False).head(5)[
        ["account_id", "total_degree", "fan_in", "fan_out"]
    ].to_dict(orient="records")
    stats["risk_score_distribution"] = edges["risk_score"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict()
    return stats


def temporal_split(edges: pd.DataFrame, day_column: str = "event_day") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset temporally into train/validation/test (60/20/20 by day index)."""

    edges = edges.copy()
    alt_col = "timestamp_day" if day_column not in edges.columns else day_column
    edges[alt_col] = edges[alt_col].fillna(0).astype(int)
    unique_days = sorted(edges[alt_col].unique())
    if not unique_days:
        return edges, edges, edges
    train_cut = max(1, int(len(unique_days) * 0.6))
    val_cut = max(train_cut + 1, int(len(unique_days) * 0.8))
    train_days = set(unique_days[:train_cut])
    val_days = set(unique_days[train_cut:val_cut])
    test_days = set(unique_days[val_cut:]) or set(unique_days[-1:])
    train_df = edges[edges[alt_col].isin(train_days)]
    val_df = edges[edges[alt_col].isin(val_days)]
    test_df = edges[edges[alt_col].isin(test_days)]
    return train_df, val_df, test_df


def build_feature_table(edges: pd.DataFrame, graph_features: pd.DataFrame) -> pd.DataFrame:
    """Join sender/receiver graph features onto the edge table."""

    sender_features = graph_features.rename(columns={"account_id": "sender_account"})
    sender_features = sender_features.rename(columns={c: f"sender_{c}" for c in sender_features.columns if c != "sender_account"})
    receiver_features = graph_features.rename(columns={"account_id": "receiver_account"})
    receiver_features = receiver_features.rename(
        columns={c: f"receiver_{c}" for c in receiver_features.columns if c != "receiver_account"}
    )
    merged = edges.merge(sender_features, on="sender_account", how="left").merge(receiver_features, on="receiver_account", how="left")
    merged = merged.fillna(0)
    return merged


def _fit_baseline(X: pd.DataFrame, y: pd.Series):
    """Train a simple baseline classifier with probability outputs."""

    if HAVE_LIGHTGBM:
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    else:
        model = GradientBoostingClassifier(random_state=0)
    model.fit(X, y)
    return model


def evaluate_model(model, splits: Dict[str, pd.DataFrame], feature_cols: Iterable[str]) -> Dict[str, float]:
    """Evaluate AUROC and F1 on validation and test splits."""

    results: Dict[str, float] = {}
    for split_name in ["val", "test"]:
        frame = splits.get(split_name)
        if frame is None or frame.empty:
            continue
        X = frame[feature_cols]
        y_true = frame["is_money_laundering"].astype(int)
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        if len(np.unique(y_true)) > 1:
            results[f"{split_name}_auroc"] = float(roc_auc_score(y_true, proba))
        else:
            results[f"{split_name}_auroc"] = float("nan")
        results[f"{split_name}_f1"] = float(f1_score(y_true, preds))
    return results


def run_quick_simulation(config_path: Path, scenario: str = "LI", days: int = 1) -> Simulation:
    """Initialise and run a short simulation for the given scenario."""

    config = load_config(config_path)
    preset = config.intensity_presets.get(scenario.upper(), {})
    config.scenario = scenario
    config.population_scale = float(preset.get("population_scale", config.population_scale))
    config.transaction_scale = float(preset.get("transaction_scale", config.transaction_scale))
    config.laundering_intensity = float(preset.get("laundering_intensity", config.laundering_intensity))
    config.simulation_days = max(1, days)
    sim = Simulation(config)
    sim.run()
    summary = sim.get_daily_summary(as_json=False)
    print(f"[{scenario}] {summary}")
    return sim


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline benchmarking for launder-simulation exports.")
    parser.add_argument("--data-dir", type=Path, default=Path("outputs"), help="Directory containing exported CSVs.")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"), help="Config path for optional simulations.")
    parser.add_argument("--simulate", action="store_true", help="Run short LI/HI simulations to illustrate scenario setup.")
    args = parser.parse_args()

    exports = load_exports(args.data_dir)
    edges = attach_risk_scores(exports["edges"])
    graph_features = exports["features"]
    stats = compute_summary_stats(edges, graph_features)
    print("\nSummary statistics:")
    for key, val in stats.items():
        print(f"- {key}: {val}")

    merged = build_feature_table(edges, graph_features)
    train_df, val_df, test_df = temporal_split(merged, day_column="event_day")
    feature_cols = [c for c in merged.columns if c.startswith(("sender_", "receiver_"))]
    if train_df.empty:
        train_df = merged
    if val_df.empty:
        val_df = train_df
    if test_df.empty:
        test_df = val_df

    model = _fit_baseline(train_df[feature_cols], train_df["is_money_laundering"].astype(int))
    metrics = evaluate_model(model, {"val": val_df, "test": test_df}, feature_cols)
    print("\nBaseline metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v:.4f}")

    if args.simulate:
        print("\nRunning illustrative LI/HI simulations...")
        for scenario in ["LI", "HI"]:
            run_quick_simulation(args.config, scenario=scenario, days=1)


if __name__ == "__main__":
    main()
