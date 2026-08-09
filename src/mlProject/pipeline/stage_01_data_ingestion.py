from mlProject.components.data_ingestion import DataIngestion
from mlProject.config.configuration import ConfigurationManager


STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(
            config=data_ingestion_config
        )

        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    print(f">>>>>> {STAGE_NAME} started <<<<<<")

    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()

    print(f">>>>>> {STAGE_NAME} completed <<<<<<")