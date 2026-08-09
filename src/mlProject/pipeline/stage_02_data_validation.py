from mlProject.components.data_validation import DataValidation
from mlProject.config.configuration import ConfigurationManager


STAGE_NAME = "Data Validation Stage"


class DataValidationTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()

        data_validation = DataValidation(
            config=data_validation_config
        )

        validation_status = data_validation.validate_all_columns()

        if not validation_status:
            raise ValueError(
                "Data validation failed. Check artifacts/data_validation/status.txt"
            )


if __name__ == "__main__":
    print(f">>>>>> {STAGE_NAME} started <<<<<<")

    pipeline = DataValidationTrainingPipeline()
    pipeline.main()

    print(f">>>>>> {STAGE_NAME} completed <<<<<<")