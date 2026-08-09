from pathlib import Path

import pandas as pd

from mlProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Validate that the dataset contains all expected columns
        defined in schema.yaml.
        """
        data_path = self.config.unzip_data_dir

        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {data_path}"
            )

        data = pd.read_csv(data_path)

        actual_columns = set(data.columns)
        expected_columns = set(self.config.all_schema.keys())

        missing_columns = expected_columns - actual_columns
        unexpected_columns = actual_columns - expected_columns

        validation_status = (
            len(missing_columns) == 0
            and len(unexpected_columns) == 0
        )

        self.config.status_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.config.status_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(f"Validation status: {validation_status}\n")

            if missing_columns:
                file.write(
                    f"Missing columns: {sorted(missing_columns)}\n"
                )

            if unexpected_columns:
                file.write(
                    f"Unexpected columns: {sorted(unexpected_columns)}\n"
                )

        return validation_status