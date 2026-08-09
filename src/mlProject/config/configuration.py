from pathlib import Path

from mlProject.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from mlProject.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
)
from mlProject.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories(
            [Path(self.config.artifacts_root)],
            verbose=False,
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories(
            [Path(config.root_dir)],
            verbose=False,
        )

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories(
            [Path(config.root_dir)],
            verbose=False,
        )

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            unzip_data_dir=Path(config.unzip_data_dir),
            status_file=Path(config.status_file),
            all_schema=dict(schema),
        )