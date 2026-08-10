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