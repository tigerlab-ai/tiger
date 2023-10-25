from tigertune.finetuning import TextGenerationTransformersFinetuneEngine
from tigertune.finetuning import TextClassificationTransformersFinetuneEngine


def classification() -> None:
    training_dataset = "/content/drive/MyDrive/tiger/TigerTune/tigertune/datasets/classification/training"
    validation_dataset = "/content/drive/MyDrive/tiger/TigerTune/tigertune/datasets/classification/validation"
    eval_dataset = "/content/drive/MyDrive/tiger/TigerTune/tigertune/datasets/classification/test_dataset.csv",

    finetune_engine = TextClassificationTransformersFinetuneEngine(
        base_model_id="distilbert-base-uncased",
    )
    finetune_engine.finetune(
        training_dataset,
        validation_dataset,
        model_output_path="exp_finetune_classification"
    )
    finetune_engine.evaluate(
        eval_dataset,
        eval_output_path="/content/drive/MyDrive/exp_finetune_classification/eval_result")


def generation() -> None:
    training_dataset = "tigertune/datasets/generation/toy_data_train.jsonl"
    eval_dataset = "tigertune/datasets/generation/toy_data_evaluation.jsonl"

    finetune_engine = TextGenerationTransformersFinetuneEngine(
        training_dataset,
        base_model_id="daryl149/llama-2-7b-chat-hf",
        eval_dataset=eval_dataset,
        model_output_path="exp_finetune"
    )
    finetune_engine.finetune()
    finetune_engine.inference(
        prompt="What is RAG?")


def main() -> None:
    print("Running TextGenerationTransformersFinetuneEngine demo:")
    generation()
    print("Running TextClassificationTransformersFinetuneEngine demo:")
    classification()


if __name__ == "__main__":
    main()
