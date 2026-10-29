import pandas as pd
import pytest

from src.data_loader import DatasetValidationError, load_dataset, validate_dataset


def test_valid_dataset_and_numeric_missing_values_are_accepted(sample_data):
    validated = validate_dataset(sample_data)
    assert validated.shape == sample_data.shape
    assert pd.isna(validated.loc[0, "TotalCharges"])


def test_missing_required_column_is_rejected(sample_data):
    with pytest.raises(DatasetValidationError, match="Missing required columns"):
        validate_dataset(sample_data.drop(columns=["Contract"]))


def test_unexpected_columns_can_be_rejected_for_strict_schema(sample_data):
    sample_data["unexpected"] = "value"
    with pytest.raises(DatasetValidationError, match="Unexpected columns"):
        validate_dataset(sample_data, allow_extra_columns=False)


def test_invalid_target_is_rejected(sample_data):
    sample_data.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DatasetValidationError, match="must contain only"):
        validate_dataset(sample_data)


def test_invalid_numeric_value_is_rejected(sample_data):
    sample_data["tenure"] = sample_data["tenure"].astype(object)
    sample_data.loc[0, "tenure"] = "unknown"
    with pytest.raises(DatasetValidationError, match="non-numeric"):
        validate_dataset(sample_data)


def test_negative_numeric_value_is_rejected(sample_data):
    sample_data.loc[0, "MonthlyCharges"] = -1
    with pytest.raises(DatasetValidationError, match="cannot contain negative"):
        validate_dataset(sample_data)


def test_unexpected_training_category_is_rejected(sample_data):
    sample_data.loc[0, "Contract"] = "Special custom plan"
    with pytest.raises(DatasetValidationError, match="unexpected categories"):
        validate_dataset(sample_data)


def test_empty_dataset_is_rejected(sample_data):
    with pytest.raises(DatasetValidationError, match="Dataset is empty"):
        validate_dataset(sample_data.iloc[0:0])


def test_empty_csv_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Could not read CSV"):
        load_dataset(path)
