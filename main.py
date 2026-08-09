from mlProject.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from mlProject.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline,
)
from mlProject.pipeline.stage_03_data_transformation import (
    DataTransformationTrainingPipeline,
)
from mlProject.pipeline.stage_04_model_trainer import (
    ModelTrainerTrainingPipeline,
)
from mlProject.pipeline.stage_05_model_evaluation import (
    ModelEvaluationTrainingPipeline,
)


def run_pipeline() -> None:
    stages = [
        ("Data Ingestion", DataIngestionTrainingPipeline()),
        ("Data Validation", DataValidationTrainingPipeline()),
        ("Data Transformation", DataTransformationTrainingPipeline()),
        ("Model Training", ModelTrainerTrainingPipeline()),
        ("Model Evaluation", ModelEvaluationTrainingPipeline()),
    ]

    for stage_name, pipeline in stages:
        print(f"\n>>>>>> {stage_name} started <<<<<<")

        try:
            pipeline.main()
        except Exception as exc:
            print(f">>>>>> {stage_name} failed <<<<<<")
            raise RuntimeError(
                f"Pipeline failed during {stage_name}"
            ) from exc

        print(f">>>>>> {stage_name} completed <<<<<<")

    print("\n>>>>>> WineMetric pipeline completed successfully <<<<<<")


if __name__ == "__main__":
    run_pipeline()