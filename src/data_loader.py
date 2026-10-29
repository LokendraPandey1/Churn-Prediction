"""Loading, validating, and cleaning the IBM Telco-style churn data."""

from pathlib import Path
import os

import numpy as np
import pandas as pd

TARGET = "Churn"
IDENTIFIER = "customerID"
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
EXPECTED_COLUMNS = [IDENTIFIER, *FEATURE_COLUMNS, TARGET]
ALLOWED_TARGETS = {"Yes", "No"}
CATEGORICAL_VALUES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot safely be used for training or scoring."""


def load_dataset(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    """Read and validate a CSV without silently downloading or inventing data."""
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Add a valid CSV or set "
            "CHURN_DATA_PATH; see data/README.md for setup instructions."
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
    """Validate schema and values, returning a cleaned copy."""
    if not isinstance(frame, pd.DataFrame):
        raise DatasetValidationError("Dataset must be a pandas DataFrame.")
    if frame.empty:
        raise DatasetValidationError("Dataset is empty; provide at least one row.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DatasetValidationError(f"Duplicate column names found: {duplicates}.")

    required = set(FEATURE_COLUMNS)
    if require_target:
        required.add(TARGET)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatasetValidationError(f"Missing required columns: {', '.join(missing)}.")
    if not allow_extra_columns:
        allowed = required | {IDENTIFIER}
        unexpected = sorted(set(frame.columns).difference(allowed))
        if unexpected:
            raise DatasetValidationError(f"Unexpected columns: {', '.join(unexpected)}.")
    if TARGET in frame.columns and not require_target:
        raise DatasetValidationError(
            f"Scoring input must not contain target column '{TARGET}'."
        )

    cleaned = frame.copy()
    for column in NUMERIC_COLUMNS:
        original = cleaned[column]
        converted = pd.to_numeric(original.replace(r"^\s*$", pd.NA, regex=True), errors="coerce")
        invalid = original.notna() & converted.isna() & original.astype(str).str.strip().ne("")
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

    senior = cleaned["SeniorCitizen"].dropna()
    if not senior.isin([0, 1]).all():
        raise DatasetValidationError("'SeniorCitizen' values must be 0, 1, or missing.")

    if require_target:
        targets = cleaned[TARGET].astype("string").str.strip()
        invalid_targets = targets.notna() & ~targets.isin(ALLOWED_TARGETS)
        if targets.isna().any() or invalid_targets.any():
            invalid_values = targets[invalid_targets].dropna().unique().tolist()
            raise DatasetValidationError(
                f"'{TARGET}' must contain only 'Yes' or 'No' and cannot be missing; "
                f"invalid values: {invalid_values[:5]}."
            )
        cleaned[TARGET] = targets

    return cleaned


def get_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the fixed prediction schema, excluding IDs, labels, and extras."""
    validate_dataset(frame, require_target=TARGET in frame.columns)
    return frame.loc[:, FEATURE_COLUMNS].copy()


def default_data_path() -> Path:
    """Resolve the configured data path, defaulting to data/telco_churn.csv."""
    configured = Path(os.environ.get("CHURN_DATA_PATH", "data/telco_churn.csv"))
    if configured.is_absolute():
        return configured
    return Path(__file__).resolve().parents[1] / configured
