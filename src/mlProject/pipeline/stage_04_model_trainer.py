from mlProject.components.model_trainer import ModelTrainer
from mlProject.config.configuration import ConfigurationManager


STAGE_NAME = "Model Trainer Stage"


class ModelTrainerTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()

        model_trainer = ModelTrainer(
            config=model_trainer_config
        )

        model_trainer.train()


if __name__ == "__main__":
    print(f">>>>>> {STAGE_NAME} started <<<<<<")

    pipeline = ModelTrainerTrainingPipeline()
    pipeline.main()

    print(f">>>>>> {STAGE_NAME} completed <<<<<<")