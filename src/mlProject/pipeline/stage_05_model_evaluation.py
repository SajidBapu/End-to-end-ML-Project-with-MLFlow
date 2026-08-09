from mlProject.components.model_evaluation import ModelEvaluation
from mlProject.config.configuration import ConfigurationManager


STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        model_evaluation = ModelEvaluation(
            config=model_evaluation_config
        )

        model_evaluation.evaluate()


if __name__ == "__main__":
    print(f">>>>>> {STAGE_NAME} started <<<<<<")

    pipeline = ModelEvaluationTrainingPipeline()
    pipeline.main()

    print(f">>>>>> {STAGE_NAME} completed <<<<<<")