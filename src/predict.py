"""Safe single-record and batch scoring helpers."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data_loader import FEATURE_COLUMNS, IDENTIFIER, TARGET, validate_dataset


class ModelArtifactError(ValueError):
    """Raised when a persisted model is missing or incompatible."""


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    model_path = Path(path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Run `python -m src.train` "
            "after configuring a valid dataset."
        )
    try:
        bundle = joblib.load(model_path)
    except (OSError, ValueError, EOFError) as exc:
        raise ModelArtifactError(f"Could not load model artifact '{model_path}': {exc}") from exc
    if not isinstance(bundle, dict) or not {"pipeline", "feature_columns", "threshold"} <= bundle.keys():
        raise ModelArtifactError("Model artifact is missing required pipeline metadata.")
    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise ModelArtifactError("Model artifact feature schema does not match this application.")
    if not hasattr(bundle["pipeline"], "predict_proba"):
        raise ModelArtifactError("Model artifact pipeline does not support probability predictions.")
    return bundle


def _validate_scoring_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if TARGET in frame.columns:
        raise ValueError(f"Scoring input must not include target column '{TARGET}'.")
    if IDENTIFIER in frame.columns:
        model_frame = frame.drop(columns=[IDENTIFIER])
    else:
        model_frame = frame
    validated = validate_dataset(model_frame, require_target=False)
    return validated.loc[:, FEATURE_COLUMNS]


def predict_customers(
    frame: pd.DataFrame,
    pipeline: Any,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score rows, preserving all supplied columns and adding probability and risk."""
    if not 0 <= threshold <= 1:
        raise ValueError("Classification threshold must be between 0 and 1.")
    features = _validate_scoring_frame(frame)
    probabilities = pipeline.predict_proba(features)[:, 1]
    result = frame.copy()
    result["churn_probability"] = probabilities
    result["risk_category"] = ["High" if value >= threshold else "Low" for value in probabilities]
    return result
