"""Leakage-safe preprocessing pipeline construction."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import FEATURE_COLUMNS, NUMERIC_COLUMNS

CATEGORICAL_COLUMNS = [column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS]


def build_preprocessor() -> ColumnTransformer:
    """Create transforms that are fitted only as part of a training pipeline."""
    numeric = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
