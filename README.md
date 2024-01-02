# Harnessing LLMs for Cheese Description Generation

This repository contains code for an E2E framework for the generation of cheese descriptions using Large Languange Models (LLMs).

## Install Requirements

`
pip install -r requirements.txt
`
## Data Synthesis

For preparation of training data, we use the OpenAI GPT-3.5 turbo model. The raw data comprising 600 cheese descriptions text files are present in `./Data/raw_data`.

### Creating pre-requisite schema definitions
We have prepared a formatted JSON file `./Data/template_english_dict.jsonl` with model lines based on the CDG tool (corresponding to all the moves and steps).


Using this as prior information about the domain, we prepare our own schema with customer slots for each rhetorical section (M1-M6), and store them in `./Data/slots_data`. The files for M1-M6 can be found in the following locations within this directory:

1. M1: `identification.txt`
2. M2: `description.txt`
3. M3: `process.txt`
4. M4: `smell_taste.txt`
5. M5: `service_suggestions.txt`
6. M6: `quality_assurance.txt`

### Invoking OpenAI APIs for Data Generation

The data synthesis pipeline follows two steps:

1. `./Notebooks/data_reallignment.ipynb`: Run this notebook to invoke OpenAI APIs and restructure raw data instances into the 6 rhetorical categories. Update file paths in the notebook for new datasets. This outputs `./Data/slots_data/rhet_data.json`

2. `./Notebooks/slot_filling.ipynb`: Run this notebook, with the output file from (1.). The notebook reads slots_data files for slot definitions, uses realigned text from (1.) and injects into slot_filling prompts. It then invokes OpenAI APIs to auto-fill the slots with values from the text. This outputs `./Data/slots_data/rhet_data_slots.json` 

The notebooks use prompt templates defined in `./Notebooks/templates.py` to create the prompts for querying purposes. 

Use `./Notebooks/data_analysis.ipynb` to analyze the prepared data and generate cleaned data after post-processing. The cleaned data is then split into train, validation and test splits using the notebook `./Notebooks/train_val_test_split.ipynb`:

1. train_data: `./Data/slots_data/rhet_data_slots_cleaned_train.json`
2. val_data: `./Data/slots_data/rhet_data_slots_cleaned_val.json`
3. test_data: `./Data/slots_data/rhet_data_slots_cleaned_test.json`

Note: Other notebooks in the `./Notebooks/` folder are experimental and can be ignored.

## Train model

`python train.py --config configs/base.json --generation "<generation_type"`

 Note: Training requires GPU support (CUDA environment). The code will not run in CPU environment or will be extremely slow. 

## Inference (Full Test Corpus)

### Greedy Decoding
`python test.py --config configs/base.json --generation "greedy"`

### Beam Search Decoding

`python test.py --config configs/base.json --generation "beam"`

### Top-K Sampling

`python test.py --config configs/base.json --generation "sampling_topk"`

### Top-P Sampling

`python test.py --config configs/base.json --generation "sampling_topp"`

### Top-P + Top-K Sampling

`python test.py --config configs/base.json --generation "sampling_topk_topp"`


### Inference (Single Input Template)

Input template based inferencing can be done by specifying the template file path under "data->test" within the configs. A sample template is already provided under `configs/cheese_template.json`. While running test specify the flag `--from_template` as shown below: 

`python test.py --config configs/base.json --generation "<generation_type>" --from_template`


## Setting config params

 - The config file `./configs/base.json` contains training and inference parameters which should be set appropriately before performing train or inference operations. The `type` parameter refers to pre-trained model config type that is defined under "model" key within the config. Currently supported pre-trained models are `bert-base-uncased`, `EleutherAI/gpt-neo-1.3B`, `t5-base` and `google/flan-t5-base`. The config can be extended when new model code is added. 

 - Training parameters, including optimizer settings, learning rate, checkpoint paths etc. are defined under "train".

 - For loading pre-trained models to be used for inferencing, set the "checkpoint" paths correctly to refer to the train "save_dir" which was used for fine-tuning.

 ## Additional Developer Notes

 For developers planning to extend the repository for generating other product descriptions or fine-tuning other pre-trained architectures, here are some details on the source code:

 - `collators` directory contains custom collators defined per model, where custom logic for batching data is defined (PyTorch definitions)

 - `dataset.py` contains pre-defined pipelines for loading prepared data and storing in-memory a data-to-text task data for same. 

 - `pipelines.py` contains model specific data pre-processing code (tokenization, attention masks etc.)

 - `loaders` contains code for loading pre-trained or fine-tuned checkpoints based on the configs. It creates the models and updates the weights from given checkpoint paths.

 - `models` contains the source code for different model architectures used within this project. Contains model forward definition. 

 - `train.py & test.py` contains training and testing code (innitialization of dataset, optimizers, models, training/inference pipelines, saving checkpoints, evaluation etc.)

 - `evaluation` contains quantitative metric definitions for evaluating model performance such as Rouge-1,2,L, Meteor, Bleu-4, BertScore etc. 



