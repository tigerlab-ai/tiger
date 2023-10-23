"""Text Generation Finetuning Engine."""
from typing import Any, Dict, Optional, Union

from tigertune.finetuning.data_types import BaseLLMFinetuneEngine


class TextGenerationTransformersFinetuneEngine(BaseLLMFinetuneEngine):
    """Text generation transformers finetune engine.

    Args:
        training_dataset (`str`): Dataset filename to finetune on.
        eval_dataset (`str`): Dataset filename for evaluation.
        base_model_id (`str`): Base model ID to finetune.
        model_output_path (`str`): Path to save model output. Defaults to "model_output".
        adapter_model (`Optional[str]`): Adapter model. Defaults to None, in which
            case QLoRA is used.
        hyperparameters (`Optional[Dict[str, Union[str, int, float]]]`): 
            A dict of hyperparameters to customize fine-tuning behavior.

            Currently supported hyperparameters:

            * `lr`: Peak learning rate used during fine-tuning. It decays with a cosine schedule afterward. (Default: 2e-3)
            * `warmup_ratio`: Ratio of training steps used for learning rate warmup. (Default: 0.03)
            * `num_train_epochs`: Number of fine-tuning epochs. This should be less than 20. (Default: 1)
            * `weight_decay`: Regularization penalty applied to learned weights. (Default: 0.001)
            * `peft_config`: A dict of parameters for the PEFT algorithm. See [LoraConfig](https://huggingface.co/docs/peft/main/en/package_reference/tuners#peft.LoraConfig) for more information. (Default: 
                LoraConfig(
                    lora_alpha=64,
                    lora_dropout=16,
                    r=0.1,
                    bias="none",
                    task_type="CAUSAL_LM",
                ))
    """

    def __init__(
        self,
        training_dataset: str,
        base_model_id: str = "meta-llama/Llama-2-7b-chat-hf",
        model_output_path: str = "finetune_result",
        eval_dataset: Optional[str] = None,
        adapter_model: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Union[str, int, float]]] = None,
    ) -> None:
        """Init params."""
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
            pipeline,
        )
        from peft import LoraConfig, PeftModel
        from trl import SFTTrainer

        train_dataset = load_dataset(
            'json', data_files=training_dataset, split='train')
        eval_dataset = load_dataset(
            'json', data_files=eval_dataset, split='train')
        self.base_model_id = base_model_id

        def formatting_func(example):
            output_texts = []
            for i in range(len(example['input'])):
                text = f"<s>[INST] {example['input'][i]}\n [/INST] {example['output'][i]}"
                if base_model_id == "tiiuae/falcon-7b":
                    text = f"### Human: {example['input'][i]} ### Assistant: {example['output'][i]}"
                output_texts.append(text)
            return output_texts

        ################################################################################
        # QLoRA parameters
        ################################################################################

        # LoRA attention dimension
        lora_r = 64

        # Alpha parameter for LoRA scaling
        lora_alpha = 16

        # Dropout probability for LoRA layers
        lora_dropout = 0.1

        ################################################################################
        # BitsAndBytes parameters
        ################################################################################

        # Activate 4-bit precision base model loading
        use_4bit = True

        # Compute dtype for 4-bit base model
        bnb_4bit_compute_dtype = "float16"

        # Quantization type (fp4 or nf4)
        bnb_4bit_quant_type = "nf4"

        # Activate nested quantization for 4-bit base models
        use_nested_quant = False

        ################################################################################
        # TrainingArguments parameters
        ################################################################################

        # Number of training epochs
        num_train_epochs = 1
        if hyperparameters is not None and hyperparameters.num_train_epochs is not None:
            num_train_epochs = hyperparameters.num_train_epochs

        # Enable fp16/bf16
        fp16 = False
        bf16 = False

        # Batch size per GPU for training
        per_device_train_batch_size = 4

        # Batch size per GPU for evaluation
        per_device_eval_batch_size = 4

        # Number of update steps to accumulate the gradients for
        gradient_accumulation_steps = 1

        # Maximum gradient normal (gradient clipping)
        max_grad_norm = 0.3

        # Weight decay to apply to all layers except bias/LayerNorm weights
        weight_decay = 0.001
        if hyperparameters is not None and hyperparameters.weight_decay is not None:
            weight_decay = hyperparameters.weight_decay

        # Optimizer to use
        optim = "paged_adamw_32bit"

        # Initial learning rate (AdamW optimizer)
        learning_rate = 2e-4
        if hyperparameters is not None and hyperparameters.learning_rate is not None:
            learning_rate = hyperparameters.learning_rate

        # Learning rate schedule
        lr_scheduler_type = "cosine"

        # Number of training steps (overrides num_train_epochs)
        max_steps = -1

        # Ratio of steps for a linear warmup (from 0 to learning rate)
        warmup_ratio = 0.03
        if hyperparameters is not None and hyperparameters.warmup_ratio is not None:
            warmup_ratio = hyperparameters.warmup_ratio

        # Group sequences into batches with same length
        # Saves memory and speeds up training considerably
        group_by_length = True

        # Save checkpoint every X updates steps
        save_steps = 0

        # Log every X updates steps
        logging_steps = 50

        # Perform evaluation at the end of training
        do_eval = True

        # Directory for storing logs
        logging_dir = "logs"

        # Save the model checkpoint every logging step
        save_strategy = "steps"
        evaluation_strategy = "steps"

        # Evaluate and save checkpoints every 50 steps
        eval_steps = 50

        ################################################################################
        # SFT parameters
        ################################################################################

        # Maximum sequence length to use
        max_seq_length = None

        # Pack multiple short examples in the same input sequence to increase efficiency
        packing = False

        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and use_4bit:
            # Check if CUDA (GPU) is available
            if torch.cuda.is_available():
                print("CUDA is available.")
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    bf16 = True
            else:
                print("CUDA is not available.")

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            # torch_dtype=torch.float32,
            # offload_folder="offload",
            # offload_state_dict=True,
            device_map={"": 0}
        )
        self.model.config.use_cache = False

        # More info: https://github.com/huggingface/transformers/pull/24906
        self.model.config.pretraining_tp = 1

        # Load LLaMA tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if hyperparameters is not None and hyperparameters.peft_config is not None:
            peft_config = hyperparameters.peft_config

        # Set training parameters
        training_arguments = TrainingArguments(
            output_dir=model_output_path,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_strategy=save_strategy,
            save_steps=save_steps,
            logging_dir=logging_dir,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            # Evaluate the model every logging step
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,               # Evaluate and save checkpoints every 50 steps
            do_eval=do_eval,                 # Perform evaluation at the end of training
            report_to="tensorboard"
        )

        # Set supervised fine-tuning parameters
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,  # hyper
            formatting_func=formatting_func,
            tokenizer=tokenizer,
            # dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=training_arguments,
            packing=packing,
        )

        self.auto_model = AutoModelForCausalLM
        self.auto_tokenizer = AutoTokenizer
        self.peft_model = PeftModel
        self.pipeline = pipeline
        self.torch = torch

        self.output_merged_dir = model_output_path
        self.base_model_id = base_model_id

    def finetune(self, **train_kwargs: Any) -> None:
        """Finetune model."""
        # Train model
        self.trainer.train()

        # Save trained model
        self.trainer.model.save_pretrained(self.output_merged_dir)

        # Empty VRAM
        del self.model
        del self.trainer
        import gc
        gc.collect()
        gc.collect()

        base_model = self.auto_model.from_pretrained(
            self.base_model_id,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=self.torch.float16,
            device_map={"": 0},
        )

        # Load the QLoRA adapter from the new model
        self.model = self.peft_model.from_pretrained(
            base_model, self.output_merged_dir)
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained(
            self.output_merged_dir, safe_serialization=True)

        # save tokenizer for easy inference
        # self.tokenizer_reload.save_pretrained(self.output_merged_dir)

    def inference(self, **model_kwargs: Any) -> str:
        """Inference."""
        # Reload tokenizer to save it
        tokenizer_reload = self.auto_tokenizer.from_pretrained(
            self.base_model_id, trust_remote_code=True)
        tokenizer_reload.pad_token = tokenizer_reload.eos_token
        tokenizer_reload.padding_side = "right"

        # Run inference
        pipe = self.pipeline(task="text-generation", model=self.model,
                             tokenizer=tokenizer_reload, max_length=200)
        base_prompt = model_kwargs["prompt"]
        prompt = f"<s>[INST] {base_prompt} [/INST]"
        if self.base_model_id == "tiiuae/falcon-7b":
            prompt = f"### Human: {model_kwargs.prompt} ### Assistant: "
        result = pipe(prompt)
        print(result[0]['generated_text'])
        return result[0]['generated_text']
