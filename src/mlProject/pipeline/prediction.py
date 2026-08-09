from pathlib import Path

import joblib
import pandas as pd


EXPECTED_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


class PredictionPipeline:
    def __init__(
        self,
        model_path: Path = Path("artifacts/model_trainer/model.joblib"),
    ):
        self.model_path = model_path

    def _validate_features(self, features: dict[str, float]) -> None:
        missing = set(EXPECTED_FEATURES) - set(features.keys())
        unexpected = set(features.keys()) - set(EXPECTED_FEATURES)

        if missing:
            raise ValueError(
                f"Missing required features: {sorted(missing)}"
            )

        if unexpected:
            raise ValueError(
                f"Unexpected features: {sorted(unexpected)}"
            )

        for feature_name, value in features.items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Feature '{feature_name}' must be numeric."
                )

    def predict(self, features: dict[str, float]) -> float:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}"
            )

        self._validate_features(features)

        model = joblib.load(self.model_path)

        input_data = pd.DataFrame(
            [[features[name] for name in EXPECTED_FEATURES]],
            columns=EXPECTED_FEATURES,
        )

        prediction = model.predict(input_data)

        return float(prediction[0])