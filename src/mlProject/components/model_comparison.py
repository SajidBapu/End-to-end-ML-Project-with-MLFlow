from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class ModelComparison:
    def __init__(
        self,
        train_data_path: Path,
        test_data_path: Path,
        target_column: str,
    ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.target_column = target_column

    @staticmethod
    def evaluate_model(actual, predicted) -> dict[str, float]:
        mse = mean_squared_error(actual, predicted)

        return {
            "rmse": float(mse ** 0.5),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)),
        }

    def compare(self) -> pd.DataFrame:
        train_data = pd.read_csv(self.train_data_path)
        test_data = pd.read_csv(self.test_data_path)

        wine_type_mapping = {
         "red": 0,
         "white": 1,
        }

        train_data["wine_type"] = train_data["wine_type"].map(
           wine_type_mapping
        )

        test_data["wine_type"] = test_data["wine_type"].map(
           wine_type_mapping
        )

        if train_data["wine_type"].isnull().any():
           raise ValueError(
           "Unknown wine_type found in training data."
        )

        if test_data["wine_type"].isnull().any():
           raise ValueError(
           "Unknown wine_type found in test data."
        )

        train_x = train_data.drop(columns=[self.target_column])
        train_y = train_data[self.target_column]

        test_x = test_data.drop(columns=[self.target_column])
        test_y = test_data[self.target_column]

        models = {
            "ElasticNet": ElasticNet(
                alpha=0.2,
                l1_ratio=0.1,
                random_state=42,
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                random_state=42,
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
        }

        results = []

        mlflow.set_experiment("WineMetric-Model-Comparison")

        for model_name, model in models.items():
            model.fit(train_x, train_y)

            predictions = model.predict(test_x)

            metrics = self.evaluate_model(
                test_y,
                predictions,
            )

            with mlflow.start_run(run_name=model_name):
                mlflow.log_param("model_name", model_name)

                mlflow.log_metrics(metrics)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                )

            results.append(
                {
                    "model": model_name,
                    **metrics,
                }
            )

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="rmse",
            ascending=True,
        ).reset_index(drop=True)

        return results_df
