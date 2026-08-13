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
    "wine_type",
]


class PredictionPipeline:
    def __init__(
        self,
        model_path: Path = Path("artifacts/model_trainer/model.joblib"),
    ):
        self.model_path = model_path

    def _validate_features(self, features: dict) -> None:
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

        # Validate numeric physicochemical features
        for feature_name in EXPECTED_FEATURES:
            if feature_name == "wine_type":
                continue

            value = features[feature_name]

            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Feature '{feature_name}' must be numeric."
                )

        # Validate wine type separately
        wine_type = features["wine_type"]

        if not isinstance(wine_type, str):
            raise TypeError(
                "Feature 'wine_type' must be either 'red' or 'white'."
            )

        if wine_type.strip().lower() not in {"red", "white"}:
            raise ValueError(
                "Feature 'wine_type' must be either 'red' or 'white'."
            )

    def predict(self, features: dict) -> float:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}"
            )

        self._validate_features(features)

        # Create a copy so the original input dictionary is not modified
        processed_features = features.copy()

        wine_type_mapping = {
            "red": 0,
            "white": 1,
        }

        processed_features["wine_type"] = wine_type_mapping[
            processed_features["wine_type"].strip().lower()
        ]

        model = joblib.load(self.model_path)

        input_data = pd.DataFrame(
            [[processed_features[name] for name in EXPECTED_FEATURES]],
            columns=EXPECTED_FEATURES,
        )

        prediction = model.predict(input_data)

        return float(prediction[0])