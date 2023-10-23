from tigertune.finetuning import TextGenerationTransformersFinetuneEngine


def main() -> None:
    training_dataset = "tigertune/datasets/toy_data_train.jsonl"
    eval_dataset = "tigertune/datasets/toy_data_evaluation.jsonl"

    finetune_engine = TextGenerationTransformersFinetuneEngine(
        training_dataset,
        base_model_id="daryl149/llama-2-7b-chat-hf",
        eval_dataset=eval_dataset,
        model_output_path="exp_finetune"
    )
    finetune_engine.finetune()
    finetune_engine.inference(
        prompt="What is RAG?")


if __name__ == "__main__":
    main()
