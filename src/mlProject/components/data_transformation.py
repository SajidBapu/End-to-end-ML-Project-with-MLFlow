from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_split_data(self) -> tuple[Path, Path]:
        """
        Read the validated dataset, split it into training and testing sets,
        and save both files under the transformation artifacts directory.
        """
        if not self.config.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.config.data_path}"
            )

        data = pd.read_csv(self.config.data_path)

        train, test = train_test_split(
            data,
            test_size=0.20,
            random_state=42,
        )

        train_path = self.config.root_dir / "train.csv"
        test_path = self.config.root_dir / "test.csv"

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        print(f"Training data saved to: {train_path}")
        print(f"Testing data saved to: {test_path}")
        print(f"Training rows: {len(train)}")
        print(f"Testing rows: {len(test)}")

        return train_path, test_path