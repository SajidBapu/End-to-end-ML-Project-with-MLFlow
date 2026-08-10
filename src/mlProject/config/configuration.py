from pathlib import Path

from mlProject.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from mlProject.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
)
from mlProject.utils.common import create_directories, read_yaml
from mlProject.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
)

from mlProject.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from mlProject.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


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

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories(
            [Path(config.root_dir)],
            verbose=False,
        )

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.ExtraTrees
        schema = self.schema.TARGET_COLUMN

        create_directories(
            [Path(config.root_dir)],
            verbose=False,
        )

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            model_name=config.model_name,
            n_estimators=int(params.n_estimators),
            random_state=int(params.random_state),
            n_jobs=int(params.n_jobs),
            target_column=schema.name,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories(
            [Path(config.root_dir)],
            verbose=False,
        )

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            metric_file_name=Path(config.metric_file_name),
            all_params=dict(self.params.ExtraTrees),
            target_column=self.schema.TARGET_COLUMN.name,
        )
