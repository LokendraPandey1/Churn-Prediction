"""Reproducible model comparison and final holdout evaluation."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import (
    FEATURE_COLUMNS,
    TARGET,
    default_data_path,
    get_features,
    load_dataset,
    validate_dataset,
)
from src.explainability import global_permutation_importance
from src.preprocessing import build_preprocessor

RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def train_and_evaluate(
    frame: pd.DataFrame,
    artifacts_dir: str | Path = "models",
    cv_folds: int = 5,
) -> dict:
    if cv_folds < 2:
        raise ValueError("Cross-validation requires at least two folds.")
    validated = validate_dataset(frame, require_target=True)
    features = get_features(validated)
    target = validated[TARGET].map({"No": 0, "Yes": 1})
    counts = target.value_counts()
    if len(counts) != 2 or counts.min() < cv_folds:
        raise ValueError(
            f"Each churn class must contain at least {cv_folds} rows for stratified "
            f"{cv_folds}-fold cross-validation; found {counts.to_dict()}."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    train_class_counts = y_train.value_counts()
    if len(train_class_counts) != 2 or train_class_counts.min() < cv_folds:
        raise ValueError(
            f"The training partition needs at least {cv_folds} examples per class "
            f"for cross-validation; found {train_class_counts.to_dict()}. "
            "Provide more labeled rows or reduce the number of folds."
        )
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scorers = {"precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    cv_results = {}
    fitted_models = {}
    for name, classifier in build_models().items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("classifier", classifier)]
        )
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=folds,
            scoring=scorers,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )
        cv_results[name] = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"])),
            }
            for metric in scorers
        }
        fitted_models[name] = pipeline.fit(x_train, y_train)

    selected_name = max(
        cv_results,
        key=lambda name: (cv_results[name]["roc_auc"]["mean"], cv_results[name]["recall"]["mean"]),
    )
    selected = fitted_models[selected_name]
    probabilities = selected.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)
    test_metrics = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "threshold": DEFAULT_THRESHOLD,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1], target_names=["No churn", "Churn"],
            output_dict=True, zero_division=0,
        ),
        "test_rows": int(len(y_test)),
    }
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    test_metrics["roc_curve"] = {
        "false_positive_rate": false_positive_rate.tolist(),
        "true_positive_rate": true_positive_rate.tolist(),
    }

    global_importance = global_permutation_importance(
        selected, x_test, y_test, random_state=RANDOM_STATE
    ).to_dict(orient="records")
    tree_model_name = max(
        ["Random Forest", "Gradient Boosting"],
        key=lambda name: cv_results[name]["roc_auc"]["mean"],
    )
    tree_model = fitted_models[tree_model_name]
    tree_classifier = tree_model.named_steps["classifier"]
    encoded_features = tree_model.named_steps["preprocessor"].get_feature_names_out()
    tree_importance = (
        pd.DataFrame(
            {
                "feature": encoded_features,
                "importance": tree_classifier.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .to_dict(orient="records")
    )
    results = {
        "selected_model": selected_name,
        "selection_metric": "highest mean cross-validation ROC-AUC (recall breaks ties)",
        "random_state": RANDOM_STATE,
        "cv_folds": cv_folds,
        "training_rows": int(len(x_train)),
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "global_permutation_importance": global_importance,
        "tree_feature_importance": tree_importance,
        "tree_importance_model": tree_model_name,
    }
    reference_values = {}
    for column in FEATURE_COLUMNS:
        if column in {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}:
            reference_values[column] = x_train[column].median()
        else:
            modes = x_train[column].mode(dropna=True)
            reference_values[column] = modes.iloc[0] if not modes.empty else "Unknown"
    reference_profile = pd.DataFrame([reference_values], columns=FEATURE_COLUMNS)

    destination = Path(artifacts_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": selected,
            "feature_columns": FEATURE_COLUMNS,
            "threshold": DEFAULT_THRESHOLD,
            "selected_model": selected_name,
            "metrics": results,
            "reference_profile": reference_profile,
        },
        destination / "churn_pipeline.joblib",
    )
    (destination / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate customer churn models.")
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models"))
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    dataset = load_dataset(args.data)
    results = train_and_evaluate(dataset, args.artifacts_dir, args.cv_folds)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
