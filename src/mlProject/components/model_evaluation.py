import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    @staticmethod
    def eval_metrics(actual, predicted) -> tuple[float, float, float]:
        mse = mean_squared_error(actual, predicted)
        rmse = mse ** 0.5
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        return rmse, mae, r2

    def evaluate(self) -> None:
        test_data = pd.read_csv(self.config.test_data_path)

        # Encode wine type exactly the same way as during training
        wine_type_mapping = {
             "red": 0,
             "white": 1
        }

        test_data["wine_type"] = test_data["wine_type"].map(wine_type_mapping)

        if test_data["wine_type"].isnull().any():
           raise ValueError("Unknown wine_type found in evaluation data.")


        model = joblib.load(self.config.model_path)

        test_x = test_data.drop(
            columns=[self.config.target_column]
        )

        test_y = test_data[self.config.target_column]

        predicted_qualities = model.predict(test_x)

        rmse, mae, r2 = self.eval_metrics(
            test_y,
            predicted_qualities,
        )

        scores = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }

        save_json(
            path=self.config.metric_file_name,
            data=scores,
        )

        mlflow.set_experiment("WineMetric")

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)

            mlflow.log_metrics(
                {
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                }
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
            )

        print("Model evaluation completed successfully.")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")
