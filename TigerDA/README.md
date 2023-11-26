# TigerDA
A package used to augment datasets.

## Preview**:
There are many data augmentation methods. This toolkit provides both pertubation-based data augmentation and generation-based data augmentation.

### Generation-based data augmentation
For generation-based data augmentation, there are normally 2 steps:
1. Fine tune the model. It is to make the model generate output with a certain text structure or focus on one theme. Normally it is not enough to just use the pre-trained model available in Transformers. This step is covered in TigerTune package.
2. Text generation with the fine tuned model. This is covered in this package.

## Installation

For development or research, you can clone and install the repo locally:
```shell
git clone https://github.com/tigerlab-ai/tiger.git && cd TigeDA
pip install --upgrade -e .
```
This will install the TigerDA repo and all necessary dependencies.

On a non-intel Mac you may need to downgrade `transformers` library: `pip install transformers==4.30`.

## Data Setup
If you do not have a dataset, you can start with ours toy data in the tigerda/datasets folder.

The setup for augmentation can be effortlessly executed provided you possess a csv file containing data entries with 1 field: `seed`.

## Usage
Initialize text generation augmenter. We use a GPT2 model finetuned on the open-instruct-v1 dataset as an example.
```python
augmentation_engine = TextGenerationDataAugmentationEngine(
    model_id="vicgalle/gpt2-open-instruct-v1",
)
```

Generate with a seed dataset
```python
augmentation_engine.augment(
    seed_dataset="datasets/seed_dataset.csv",
    number_of_rows=1,
    num_return_sequences=1,
    output_path="datasets/results.json"
)
```

You can pass in optional formatting_func too, for different generation models.
```python
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
```

## Demo
```shell
python3 examples/text_generation_augmenter_example.py 
```


