from numbers import Real

import pytest

from mlProject.pipeline.prediction import PredictionPipeline


VALID_FEATURES = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "wine_type": "red",
}


def test_prediction_returns_numeric_value():
    pipeline = PredictionPipeline()

    prediction = pipeline.predict(VALID_FEATURES)

    assert isinstance(prediction, Real)


def test_prediction_rejects_missing_features():
    pipeline = PredictionPipeline()

    invalid_features = {
        "alcohol": 9.4,
    }

    with pytest.raises(ValueError, match="Missing required features"):
        pipeline.predict(invalid_features)


def test_prediction_rejects_unexpected_features():
    pipeline = PredictionPipeline()

    invalid_features = VALID_FEATURES.copy()
    invalid_features["unknown feature"] = 123.0

    with pytest.raises(ValueError, match="Unexpected features"):
        pipeline.predict(invalid_features)


def test_prediction_rejects_non_numeric_value():
    pipeline = PredictionPipeline()

    invalid_features = VALID_FEATURES.copy()
    invalid_features["alcohol"] = "invalid"

    with pytest.raises(TypeError, match="must be numeric"):
        pipeline.predict(invalid_features)