"""Reusable exploratory analysis for the Telco churn dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data_loader import FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET, default_data_path, load_dataset

SEGMENT_COLUMNS = [
    "Contract Length",
    "Tenure",
    "Subscription Type",
    "Gender",
    "Support Calls",
    "Payment Delay",
]


def churn_summary(frame: pd.DataFrame) -> dict:
    churned = int(frame[TARGET].eq(1).sum())
    return {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "schema": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "customers": int(len(frame)),
        "churned": churned,
        "retained": int(len(frame) - churned),
        "churn_rate": float(churned / len(frame)),
        "class_distribution": {
            "Churned": {"count": churned, "proportion": float(churned / len(frame))},
            "Retained": {
                "count": int(len(frame) - churned),
                "proportion": float((len(frame) - churned) / len(frame)),
            },
        },
        "duplicates": int(frame.duplicated().sum()),
        "missing_values": {
            column: int(count)
            for column, count in frame.loc[:, FEATURE_COLUMNS].isna().sum().items()
        },
        "descriptive_statistics": frame.loc[:, FEATURE_COLUMNS].describe(
            include="all"
        ).fillna("").to_dict(),
    }


def build_eda_report(frame: pd.DataFrame) -> dict:
    """Build a reproducible, data-derived report for the application/CLI."""
    segment_rates = {}
    for column in SEGMENT_COLUMNS:
        rates = churn_rate_by(frame, column)
        segment_rates[column] = [
            {
                "segment": None if pd.isna(row[column]) else str(row[column]),
                "customers": int(row["customers"]),
                "churn_rate": float(row["churn_rate"]),
            }
            for _, row in rates.iterrows()
        ]
    return {
        "summary": churn_summary(frame),
        "segment_churn_rates": segment_rates,
        "numeric_correlations": numeric_correlations(frame).to_dict(),
    }


def churn_rate_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in SEGMENT_COLUMNS:
        raise ValueError(f"Unsupported churn segment: '{column}'.")
    data = frame.assign(_churn=frame[TARGET].eq(1).astype(int))
    if column == "Tenure":
        labels = ["1-12", "13-24", "25-36", "37-48", "49+"]
        breaks = [0, 12, 24, 36, 48, float("inf")]
        data[column] = pd.cut(data[column], bins=breaks, labels=labels, include_lowest=True)
    elif column == "Payment Delay":
        labels = ["0-5", "6-10", "11-20", "21-30", "31+"]
        breaks = [-1, 5, 10, 20, 30, float("inf")]
        data[column] = pd.cut(data[column], bins=breaks, labels=labels, include_lowest=True)
    elif column == "Support Calls":
        labels = ["0-2", "3-5", "6-8", "9+"]
        breaks = [-1, 2, 5, 8, float("inf")]
        data[column] = pd.cut(data[column], bins=breaks, labels=labels, include_lowest=True)
    return (
        data.groupby(column, dropna=False, observed=False)["_churn"]
        .agg(customers="size", churn_rate="mean")
        .reset_index()
    )


def numeric_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.loc[:, NUMERIC_COLUMNS].copy()
    numeric[TARGET] = frame[TARGET].astype(int)
    return numeric.corr(numeric_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an exploratory churn analysis report.")
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument("--output", type=Path, default=Path("reports/eda_summary.json"))
    args = parser.parse_args()
    report = build_eda_report(load_dataset(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"EDA report written to {args.output}")


if __name__ == "__main__":
    main()
