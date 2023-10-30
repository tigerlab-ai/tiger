# TigerTune
A package used to fine tune a model.

## Usage
To perform training/fine-tuning and evaluation for LLM models on Google colab, please follow the instruction at notebooks/finetune_text_generation_model.ipynb and notebooks/finetune_text_classification_model.ipynb.

To perform training/fine-tuning and evaluation for LLM models locally, please follow the following steps.

## Installation

For development or research, you can clone and install the repo locally:
```shell
git clone https://github.com/tigerlab-ai/tiger.git && cd TigerTune
pip install -r tigertune/requirements.txt
pip install --upgrade -e .
```
This will install the TigerTune repo and all necessary dependencies.

On an non-intel Mac you may need to downgrade `transformers` library: `pip install transformers==4.30`.

## Data Setup
If you do not have a dataset, you can start with ours toy data in the tigertune/datasets folder.

### Text Generation
The setup for training and evaluation can be effortlessly executed provided you possess a jsonl file containing data entries with two fields: `Input`, `Output`. 

### Text Classification
For training dataset, the input csv dataset is key'ed with `comment_text`, while the output csv dataset is key'ed with `isToxic`.
The same for validation dataset and test dataset.
To note, we splitted the input and output into 2 separate files for training and validation dataset to make it easier to be consumed in code.

## Training

You can leverage our example scripts directly if you'd like.

### Train TextGeneration LLM
Fine tune `meta-llama/Llama-2-7b-chat-hf` and `distilbert-base-uncased` model.
```shell
python3 tigertune/demo/example.py 
```

## Evaluation
Once instantialized, you can run inference on the fine tuned text generation model
```shell
finetune_engine.inference(
        prompt="What is RAG?")
```

You can also run evals on the fine tuned text classification model with your test dataset.
```shell
finetune_engine.evaluate(
        eval_dataset,
        eval_output_path="/content/drive/MyDrive/exp_finetune_classification/eval_result")
```
