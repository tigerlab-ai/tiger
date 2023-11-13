"""Text Generation Data Augmentation Engine."""
from typing import Any

from tigerda.data_types import BaseDataAugmentationEngine
from tigerda.base.loaders import TigerDADataFrameLoader

import json


class TextGenerationDataAugmentationEngine(BaseDataAugmentationEngine):
    """Text generation transformers data augmentation engine.

    Args:
        model_id (`str`): Model ID for data augmentation.
    """

    def __init__(
        self,
        model_id: str = "vicgalle/gpt2-open-instruct-v1",
        prompt_template: str = "",
    ) -> None:
        """Init params."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        ################################################################################
        # Load model and tokenizer
        ################################################################################
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

    def formatting_func(self, seed):
        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {seed}

                ### Response:"""
        return text

    def augment(self, **generate_kwargs: Any) -> None:
        """ Augment data.
        Args:
            seed_dataset (`str`): Seed dataset filename to augment data on.
            number_of_rows (`int`): Number of seed data to augment.
            num_return_sequences (`int`): Number of return augmented sequences per seed data.
            output_path (`str`): Path to save output.
            hyperparameters (`Optional[Dict[str, Union[str, int, float, bool]]]`):
                A dict of hyperparameters to customize augment behavior.

                Currently supported hyperparameters:

                * `max_length` (`int`): Max number of words to tokenize in a given text. (Default: 200)
                * `no_repeat_ngram_size` (`int`): Prevent n-grams of the desired length from being repeated, to avoid repetitions of the same text. (Default: 2)
                * `repetition_penalty` (`float`): The parameter for repetition penalty. 1.0 means no penalty. (Default: 1.5)
                * `top_p` (`float`): Top-p (nucleus) sampling. If p were 0.8, the next word is selected randomly based on the probability distribution conditioned by the previous word among the set of words that add a probability greater than or equal to 0.8. (Default: 0.85)
                * `top_k` (`int`): Top-K Sampling. If K were 6, the next word would be chosen randomly between the next 6 words with the highest probability. (Default: 50)
                * `temperature` (`float`): The temperature of the distribution, to increase the probability of extracting a word from among the most probable. (Default: 0.85)
                * `do_sample` (`bool`): Needs to set to True for Top-K-Top-N sampling. (Default: True)
                * `early_stopping` (`bool`): Controls the stopping condition for beam-based methods, like beam-search. (Default: True)
        """

        ################################################################################
        # Load parameters, and check required parameters
        ################################################################################
        if 'seed_dataset' not in generate_kwargs:
            raise ValueError("Required parameter 'seed_dataset' is missing.")
        self.params = {'max_length': 200,
                       'no_repeat_ngram_size': 2,
                       'repetition_penalty': 1.5,
                       'top_p': 0.85,
                       'top_k': 50,
                       'temperature': 0.85,
                       'do_sample': True,
                       'early_stopping': True,
                       }
        if 'hyperparameters' in generate_kwargs and generate_kwargs['hyperparameters'] is not None:
            self.params.update(generate_kwargs['hyperparameters'])

        number_return_sequences = 1
        if 'num_return_sequences' in generate_kwargs and generate_kwargs['num_return_sequences'] is not None:
            number_return_sequences = generate_kwargs['num_return_sequences']

        number_of_rows = 1
        if 'number_of_rows' in generate_kwargs and generate_kwargs['number_of_rows'] is not None:
            number_of_rows = generate_kwargs['number_of_rows']

        if 'formatting_func' in generate_kwargs and generate_kwargs['formatting_func'] is not None:
            self.formatting_func = generate_kwargs['formatting_func']

        ################################################################################
        # Load seed dataset
        ################################################################################
        tda_df_loader = TigerDADataFrameLoader()
        seed_dataset_df = tda_df_loader.from_csv(
            generate_kwargs['seed_dataset'])

        model = self.model.to(self.device)

        ################################################################################
        # Generate data
        ################################################################################
        results = []
        for i in range(number_of_rows):
            formatted_seed_data = self.formatting_func(
                seed_dataset_df["seed"][i])
            text_ids = self.tokenizer.encode(
                formatted_seed_data, return_tensors='pt')
            text_ids = text_ids.to(self.device)
            generated_text_samples = model.generate(
                text_ids,
                max_length=self.params['max_length'],
                num_return_sequences=number_return_sequences,
                no_repeat_ngram_size=self.params['no_repeat_ngram_size'],
                repetition_penalty=self.params['repetition_penalty'],
                top_p=self.params['top_p'],
                top_k=self.params['top_k'],
                temperature=self.params['temperature'],
                do_sample=self.params['do_sample'],
                early_stopping=self.params['early_stopping'],
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_texts = []
            for t in generated_text_samples:
                text = self.tokenizer.decode(t, skip_special_tokens=True)
                generated_texts.append(text)

            result_entry = {
                "seed": seed_dataset_df["seed"][i],
                # "instruction": seed_dataset_df[i]["instruction"],
                "generated_result": generated_texts
            }
            print(result_entry)
            if 'output_path' in generate_kwargs and generate_kwargs['output_path'] is not None:
                results.append(result_entry)

        ################################################################################
        # Save output
        ################################################################################
        if 'output_path' in generate_kwargs and generate_kwargs['output_path'] is not None:
            with open(generate_kwargs['output_path'], "w") as file:
                json.dump(
                    results, file, indent=4, ensure_ascii=False)
