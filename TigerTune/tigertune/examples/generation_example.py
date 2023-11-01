from tigertune.finetuning import TextGenerationTransformersFinetuneEngine

# To note: CUDA GPU is needed to run this example.


def generation() -> None:
    training_dataset = "tigertune/datasets/generation/toy_data_train.jsonl"
    eval_dataset = "tigertune/datasets/generation/toy_data_evaluation.jsonl"

    finetune_engine = TextGenerationTransformersFinetuneEngine(
        training_dataset=training_dataset,
        base_model_id="daryl149/llama-2-7b-chat-hf",
        eval_dataset=eval_dataset,
        model_output_path="exp_finetune_generation"
    )
    finetune_engine.finetune()
    finetune_engine.inference(
        prompt="What is RAG?")


def main() -> None:
    print("Running TextGenerationTransformersFinetuneEngine demo:")
    generation()


if __name__ == "__main__":
    main()
