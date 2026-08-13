import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self) -> None:
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Encode wine type for machine learning
        wine_type_mapping = {
          "red": 0,
          "white": 1
        }

        train_data["wine_type"] = train_data["wine_type"].map(wine_type_mapping)
        test_data["wine_type"] = test_data["wine_type"].map(wine_type_mapping)

        if train_data["wine_type"].isnull().any():
           raise ValueError("Unknown wine_type found in training data.")

        if test_data["wine_type"].isnull().any():
           raise ValueError("Unknown wine_type found in test data.")


        train_x = train_data.drop(
            columns=[self.config.target_column]
        )
        train_y = train_data[self.config.target_column]

        test_x = test_data.drop(
            columns=[self.config.target_column]
        )

        model = ExtraTreesRegressor(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        model.fit(train_x, train_y)

        model_path = (
            self.config.root_dir
            / self.config.model_name
        )

        joblib.dump(model, model_path)

        print("ExtraTrees model trained successfully.")
        print(f"Model saved to: {model_path}")
        print(f"Training rows: {len(train_x)}")
        print(f"Testing rows: {len(test_x)}")
