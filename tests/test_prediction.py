import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_loader import FEATURE_COLUMNS, validate_dataset
from src.predict import predict_customers
from src.preprocessing import build_preprocessor


@pytest.fixture
def fitted_pipeline(sample_data):
    frame = validate_dataset(sample_data)
    x = frame.loc[:, FEATURE_COLUMNS]
    y = frame["Churn"].eq("Yes").astype(int)
    return Pipeline(
        [("preprocessor", build_preprocessor()), ("classifier", LogisticRegression(max_iter=500))]
    ).fit(x, y)


def test_batch_scoring_preserves_records_and_adds_risk(sample_data, fitted_pipeline):
    source = sample_data.drop(columns=["Churn"])
    scored = predict_customers(source, fitted_pipeline, threshold=0.5)
    assert "customerID" in scored
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["risk_category"]) <= {"High", "Low"}
    assert len(scored) == len(source)


def test_scoring_rejects_target_column(sample_data, fitted_pipeline):
    with pytest.raises(ValueError, match="must not include target"):
        predict_customers(sample_data, fitted_pipeline)


def test_scoring_rejects_missing_input(sample_data, fitted_pipeline):
    with pytest.raises(ValueError, match="Missing required columns"):
        predict_customers(sample_data.drop(columns=["Churn", "Contract"]), fitted_pipeline)


def test_scoring_rejects_empty_batch(fitted_pipeline, feature_frame):
    with pytest.raises(ValueError, match="Dataset is empty"):
        predict_customers(feature_frame.iloc[0:0], fitted_pipeline)


def test_scoring_rejects_invalid_threshold(fitted_pipeline, feature_frame):
    with pytest.raises(ValueError, match="between 0 and 1"):
        predict_customers(feature_frame, fitted_pipeline, threshold=1.1)


def test_training_and_prediction_schema_are_consistent(sample_data, fitted_pipeline):
    scored = predict_customers(sample_data.drop(columns=["Churn"]), fitted_pipeline)
    assert list(scored.columns[-2:]) == ["churn_probability", "risk_category"]
