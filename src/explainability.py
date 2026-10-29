"""Model explanations that describe associations, not causal effects."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data_loader import FEATURE_COLUMNS


def global_permutation_importance(
    pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline,
        features.loc[:, FEATURE_COLUMNS],
        target,
        scoring="roc_auc",
        n_repeats=5,
        random_state=random_state,
        n_jobs=1,
    )
    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def local_feature_effects(pipeline, record: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Estimate per-feature probability shifts against a typical reference profile."""
    row = record.loc[:, FEATURE_COLUMNS].iloc[[0]].copy()
    baseline = reference.loc[:, FEATURE_COLUMNS].iloc[[0]].copy()
    baseline_probability = float(pipeline.predict_proba(row)[:, 1][0])
    changes = []
    for feature in FEATURE_COLUMNS:
        counterfactual = row.copy()
        counterfactual.loc[:, feature] = baseline.iloc[0][feature]
        changed_probability = float(pipeline.predict_proba(counterfactual)[:, 1][0])
        changes.append(
            {
                "feature": feature,
                "probability_change": baseline_probability - changed_probability,
            }
        )
    return (
        pd.DataFrame(changes)
        .assign(absolute_change=lambda values: np.abs(values["probability_change"]))
        .sort_values("absolute_change", ascending=False)
        .reset_index(drop=True)
    )
