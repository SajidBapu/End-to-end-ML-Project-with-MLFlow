from pathlib import Path

from mlProject.components.model_comparison import ModelComparison


def main() -> None:
    comparison = ModelComparison(
        train_data_path=Path(
            "artifacts/data_transformation/train.csv"
        ),
        test_data_path=Path(
            "artifacts/data_transformation/test.csv"
        ),
        target_column="quality",
    )

    results = comparison.compare()

    print("\nModel Comparison Results")
    print("=" * 60)
    print(results.to_string(index=False))
    print("=" * 60)

    best_model = results.iloc[0]

    print("\nBest model:")
    print(f"Model: {best_model['model']}")
    print(f"RMSE: {best_model['rmse']:.4f}")
    print(f"MAE: {best_model['mae']:.4f}")
    print(f"R2: {best_model['r2']:.4f}")


if __name__ == "__main__":
    main()