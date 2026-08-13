from pathlib import Path
import shutil

import pandas as pd

from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def prepare_train_test_data(self) -> tuple[Path, Path]:
        """
        Prepare the pre-defined training and real-only test datasets.

        The split has already been created before pipeline execution.
        This prevents synthetic training samples from leaking into the
        final real evaluation set.
        """

        train_source = self.config.source_train_data_path
        test_source = self.config.source_test_data_path

        if not train_source.exists():
            raise FileNotFoundError(
                f"Training dataset not found: {train_source}"
            )

        if not test_source.exists():
            raise FileNotFoundError(
                f"Test dataset not found: {test_source}"
            )

        train_data = pd.read_csv(train_source)
        test_data = pd.read_csv(test_source)

        # Confirm train and test contain the same columns
        if list(train_data.columns) != list(test_data.columns):
            raise ValueError(
                "Training and testing datasets do not have matching columns."
            )

        # Verify expected row counts
        if len(train_data) != 43700:
            raise ValueError(
                f"Unexpected training row count: {len(train_data)}. "
                "Expected 43700."
            )

        if len(test_data) != 1300:
            raise ValueError(
                f"Unexpected test row count: {len(test_data)}. "
                "Expected 1300."
            )

        self.config.root_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_path = self.config.root_dir / "train.csv"
        test_path = self.config.root_dir / "test.csv"

        shutil.copy2(
            train_source,
            train_path,
        )

        shutil.copy2(
            test_source,
            test_path,
        )

        print(f"Training data prepared: {train_path}")
        print(f"Testing data prepared: {test_path}")
        print(f"Training rows: {len(train_data)}")
        print(f"Testing rows: {len(test_data)}")

        return train_path, test_path