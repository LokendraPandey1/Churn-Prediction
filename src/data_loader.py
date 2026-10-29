"""Loading, validating, and cleaning the supplied customer churn datasets."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Churn"
IDENTIFIER = "CustomerID"
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
]
NUMERIC_COLUMNS = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]
CATEGORICAL_COLUMNS = ["Gender", "Subscription Type", "Contract Length"]
EXPECTED_COLUMNS = [IDENTIFIER, *FEATURE_COLUMNS, TARGET]
ALLOWED_TARGETS = {0, 1}
CATEGORICAL_VALUES = {
    "Gender": {"Female", "Male"},
    "Subscription Type": {"Basic", "Standard", "Premium"},
    "Contract Length": {"Monthly", "Quarterly", "Annual"},
}
DEFAULT_TRAIN_DATA = "customer_churn_dataset-training-master.csv"
DEFAULT_TEST_DATA = "customer_churn_dataset-testing-master.csv"


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot safely be used for training or scoring."""


def load_dataset(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    """Read and validate a CSV, ignoring wholly blank lines/rows."""
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Add the expected dataset files to "
            "data/ or set CHURN_DATA_PATH; see data/README.md."
        )
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DatasetValidationError(f"Could not read CSV '{csv_path}': {exc}") from exc
    return validate_dataset(frame, require_target=require_target)


def validate_dataset(
    frame: pd.DataFrame,
    require_target: bool = True,
    allow_extra_columns: bool = True,
) -> pd.DataFrame:
    """Validate schema and values, dropping only wholly empty rows."""
    if not isinstance(frame, pd.DataFrame):
        raise DatasetValidationError("Dataset must be a pandas DataFrame.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DatasetValidationError(f"Duplicate column names found: {duplicates}.")

    cleaned = frame.dropna(how="all").copy()
    if cleaned.empty:
        raise DatasetValidationError("Dataset is empty; provide at least one populated row.")

    required = set(FEATURE_COLUMNS)
    if require_target:
        required.add(TARGET)
    missing = sorted(required.difference(cleaned.columns))
    if missing:
        raise DatasetValidationError(f"Missing required columns: {', '.join(missing)}.")
    if not allow_extra_columns:
        allowed = required | {IDENTIFIER}
        unexpected = sorted(set(cleaned.columns).difference(allowed))
        if unexpected:
            raise DatasetValidationError(f"Unexpected columns: {', '.join(unexpected)}.")
    if TARGET in cleaned.columns and not require_target:
        raise DatasetValidationError(
            f"Scoring input must not contain target column '{TARGET}'."
        )

    for column in NUMERIC_COLUMNS:
        original = cleaned[column]
        converted = pd.to_numeric(
            original.replace(r"^\s*$", pd.NA, regex=True), errors="coerce"
        )
        invalid = (
            original.notna()
            & converted.isna()
            & original.astype(str).str.strip().ne("")
        )
        if invalid.any():
            raise DatasetValidationError(
                f"Column '{column}' contains non-numeric values; examples: "
                f"{original[invalid].astype(str).head(3).tolist()}."
            )
        cleaned[column] = converted
        present = cleaned[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(present).all():
            raise DatasetValidationError(f"Column '{column}' must contain finite values.")
        if cleaned[column].lt(0).any():
            raise DatasetValidationError(f"Column '{column}' cannot contain negative values.")

    for column, allowed in CATEGORICAL_VALUES.items():
        values = cleaned[column].astype("string").str.strip().replace("", pd.NA)
        if require_target:
            unexpected = sorted(set(values.dropna().unique()).difference(allowed))
            if unexpected:
                raise DatasetValidationError(
                    f"Column '{column}' contains unexpected categories: {unexpected[:5]}."
                )
        cleaned[column] = values

    if require_target:
        target = pd.to_numeric(cleaned[TARGET], errors="coerce")
        invalid = target.isna() | ~target.isin(ALLOWED_TARGETS)
        if invalid.any():
            values = cleaned.loc[invalid, TARGET].astype(str).unique().tolist()
            raise DatasetValidationError(
                f"'{TARGET}' must contain only numeric 0/1 values and cannot be missing; "
                f"invalid values: {values[:5]}."
            )
        cleaned[TARGET] = target.astype("int8")

    return cleaned


def get_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select model features while excluding customer IDs and labels."""
    validate_dataset(frame, require_target=TARGET in frame.columns)
    return frame.loc[:, FEATURE_COLUMNS].copy()


def default_data_path() -> Path:
    """Resolve the configured training CSV or its default under data/."""
    configured_value = os.environ.get("CHURN_DATA_PATH")
    if configured_value is None:
        return Path(__file__).resolve().parents[1] / "data" / DEFAULT_TRAIN_DATA
    configured = Path(configured_value).expanduser()
    if configured.is_absolute():
        return configured
    return Path(__file__).resolve().parents[1] / configured


def default_test_data_path() -> Path:
    """Resolve the configured test CSV or the supplied test file under data/."""
    configured_value = os.environ.get("CHURN_TEST_DATA_PATH")
    if configured_value is None:
        return Path(__file__).resolve().parents[1] / "data" / DEFAULT_TEST_DATA
    configured = Path(configured_value).expanduser()
    if configured.is_absolute():
        return configured
    return Path(__file__).resolve().parents[1] / "data" / configured
