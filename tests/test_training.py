import json

from src.predict import load_model_bundle
from src.train import train_and_evaluate


def test_training_compares_models_and_saves_reloadable_artifacts(sample_data, tmp_path):
    result = train_and_evaluate(sample_data, tmp_path, cv_folds=2)
    assert len(result["cv_results"]) == 3
    assert result["test_metrics"]["test_rows"] > 0
    assert result["test_metrics"]["roc_curve"]["false_positive_rate"]
    assert result["global_permutation_importance"]
    assert result["tree_feature_importance"]
    assert result["tree_importance_model"] in {"Random Forest", "Gradient Boosting"}
    artifact_path = tmp_path / "churn_pipeline.joblib"
    assert artifact_path.is_file()
    assert load_model_bundle(artifact_path)["pipeline"] is not None
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["selected_model"] == result["selected_model"]


def test_supplied_test_frame_is_used_as_untouched_holdout(sample_data, tmp_path):
    result = train_and_evaluate(
        sample_data,
        tmp_path,
        cv_folds=2,
        test_frame=sample_data.iloc[:20].copy(),
    )
    assert result["test_source"] == "separate supplied testing CSV"
    assert result["training_rows"] == len(sample_data)
    assert result["test_metrics"]["test_rows"] == 20
    assert set(result["test_results"]) == {
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    }
