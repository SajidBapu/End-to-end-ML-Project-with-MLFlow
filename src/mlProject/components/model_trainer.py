import joblib
import pandas as pd
from sklearn.linear_model import ElasticNet

from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self) -> None:
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop(columns=[self.config.target_column])
        train_y = train_data[self.config.target_column]

        test_x = test_data.drop(columns=[self.config.target_column])
        test_y = test_data[self.config.target_column]

        model = ElasticNet(
            alpha=self.config.alpha,
            l1_ratio=self.config.l1_ratio,
            random_state=42,
        )

        model.fit(train_x, train_y)

        model_path = self.config.root_dir / self.config.model_name
        joblib.dump(model, model_path)

        print(f"Model trained successfully.")
        print(f"Model saved to: {model_path}")
        print(f"Training rows: {len(train_x)}")
        print(f"Testing rows: {len(test_x)}")
