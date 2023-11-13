from tigerda.augmenters import TextGenerationDataAugmentationEngine


def augmentation() -> None:
    # Example
    augmentation_engine = TextGenerationDataAugmentationEngine(
        model_id="vicgalle/gpt2-open-instruct-v1",
    )
    augmentation_engine.augment(
        seed_dataset="datasets/seed_dataset.csv",
        number_of_rows=1,
        num_return_sequences=1,
        output_path="results.json"
    )

    # Example with formatting_func

    def formatting_func(seed):
        text = f"""Help me augment the following query: {seed}"""
        return text

    augmentation_engine.augment(
        seed_dataset="datasets/seed_dataset.csv",
        number_of_rows=1,
        num_return_sequences=1,
        output_path="datasets/results.json",
        formatting_func=formatting_func
    )


def main() -> None:
    print("Running TextGenerationDataAugmentationEngine demo:")
    augmentation()


if __name__ == "__main__":
    main()
