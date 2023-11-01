from tigertune.finetuning import TextClassificationTransformersFinetuneEngine


def classification() -> None:
    training_dataset = "tigertune/datasets/classification/training"
    validation_dataset = "tigertune/datasets/classification/validation"
    eval_dataset = "tigertune/datasets/classification/test_dataset.csv",

    finetune_engine = TextClassificationTransformersFinetuneEngine(
        base_model_id="distilbert-base-uncased",
    )
    finetune_engine.finetune(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model_output_path="exp_finetune_classification"
    )
    finetune_engine.evaluate(
        eval_dataset,
        eval_output_path="exp_finetune_classification/eval_result")


def main() -> None:
    print("Running TextClassificationTransformersFinetuneEngine demo:")
    classification()


if __name__ == "__main__":
    main()
