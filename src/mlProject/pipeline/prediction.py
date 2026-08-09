from pathlib import Path

import joblib
import pandas as pd


class PredictionPipeline:
    def __init__(
        self,
        model_path: Path = Path("artifacts/model_trainer/model.joblib"),
    ):
        self.model_path = model_path

    def predict(self, features: dict[str, float]) -> float:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}"
            )

        model = joblib.load(self.model_path)

        input_data = pd.DataFrame([features])

        prediction = model.predict(input_data)

        return float(prediction[0])