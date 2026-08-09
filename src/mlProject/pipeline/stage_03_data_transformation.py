from mlProject.components.data_transformation import DataTransformation
from mlProject.config.configuration import ConfigurationManager


STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()

        data_transformation = DataTransformation(
            config=data_transformation_config
        )

        data_transformation.train_test_split_data()


if __name__ == "__main__":
    print(f">>>>>> {STAGE_NAME} started <<<<<<")

    pipeline = DataTransformationTrainingPipeline()
    pipeline.main()

    print(f">>>>>> {STAGE_NAME} completed <<<<<<")