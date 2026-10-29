import numpy as np

from src.data_loader import TARGET, get_features, validate_dataset
from src.preprocessing import build_preprocessor


def test_preprocessor_fits_and_keeps_fixed_feature_count(sample_data):
    frame = validate_dataset(sample_data)
    features = get_features(frame)
    transformed = build_preprocessor().fit_transform(features)
    assert transformed.shape[0] == len(frame)
    assert transformed.shape[1] > features.shape[1]
    assert np.isfinite(transformed).all()


def test_features_exclude_identifier_and_target(sample_data):
    features = get_features(sample_data)
    assert "customerID" not in features
    assert TARGET not in features
