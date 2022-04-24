# BatteryBERT

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](https://github.com/shuhuang/batterybert/blob/master/LICENSE)

BatteryBERT: A Pre-trained Language Model for Battery Database Enhancement

## Features

- BatteryBERT Pre-training: Pre-training from Scratch or Further Training
- BatteryBERT Fine-tuning: Sequence Classification + Question Answering
- BatteryBERT Usage: Document Classifier, Device Data Extractor, General Q&A Agent
- Large-scale Device Data Extraction and Database Enhancement

## Installation
Run the following commands to clone the repository and install batterybert:
```shell
git clone https://github.com/ShuHuang/batterybert.git
cd batterybert; pip install -r requirements.txt; python setup.py develop
```

## Usage
### BatteryBERT Pre-training
#### Run pre-training:

Pre-training from scratch or further training using a masked language modeling (MLM) loss.  See `python run_pretraining.py --help` for a full list of arguments and their defaults.
```shell
python run_pretrain.py \
    --train_root $TRAIN_ROOT \
    --eval_root $EVAL_ROOT \
    --output_dir $OUTPUT_DIR \
    --tokenizer_root $TOEKNIZER_ROOT \
    --checkpoint=$CHECKPOINT_DIR 
```
#### Create a new WordPiece tokenizer:

Train a WordPiece tokenizer from scratch using the dataset from `$TRAIN_ROOT`. See `python run_tokenizer.py --help` for a full list of arguments and their defaults.
```shell
python run_tokenizer.py \
    --train_root $TRAIN_DIR \
    --save_root $SAVE_DIR \
    --save_name $TOKENIZER_NAME
```
### BatteryBERT Fine-tuning
#### Run fine-tuning (question answering):

Fine-tune a BERT model on a question answering dataset (e.g. SQuAD). See `python run_finetune_qa.py --help` for a full list of arguments and their defaults.
```shell
python run_finetune_qa.py 
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR 
```

#### Run fine-tuning (document classification):

Fine-tune a BERT model on a sequence classification dataset (e.g. paper corpus). See `python run_finetune_doc_classify.py --help` for a full list of arguments and their defaults.
```shell
$ python run_finetune_doc_classify.py 
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --train_root $TRAIN_ROOT \
    --eval_root $EVAL_ROOT
```

### BatteryBERT Usage
#### Use the battery paper classifier:
```python
>>> from batterybert.apps import DocClassifier

# Model name to be changed after published
# Create a binary classifier
>>> model_name = "batterydata/test4"
>>> sample_text = "sample text"
>>> classifier = DocClassifier(model_name)

# 0 for non-battery text, 1 for battery text
>>> category = classifier.classify(sample_text)
>>> print(category)

0
```

#### Use the device data extractor:
```python
>>> from batterybert.apps import DeviceDataExtractor

# Model name to be changed after published
# Create a device data extractor
>>> model_name = "batterydata/test1"
>>> sample_text = "The anode of this Li-ion battery is graphite."
>>> extractor = DeviceDataExtractor(model_name)

# Set the confidence score threshold
>>> result = extractor.extract(sample_text, threshold=0.1)
>>> print(result)

[{'type': 'anode', 'answer': 'graphite', 'score': 0.9736555218696594, 'context': 'The anode of this battery is graphite.'}]
```

#### Use the general Q&A agent:
```python
>>> from batterybert.apps import QAAgent

# Model name to be changed after published
# Create a QA agent
>>> model_name = "batterydata/test1"
>>> context = "The University of Cambridge is a collegiate research university in Cambridge, United Kingdom. Founded in 1209 and granted a royal charter by Henry III in 1231, Cambridge is the second-oldest university in the English-speaking world and the world's fourth-oldest surviving university."
>>> question = "When was University of Cambridge founded?"
>>> qa_agent = QAAgent(model_name)

# Set the confidence score threshold
>>> result = qa_agent.answer(question=question, context=context, threshold=0.1)
>>> print(result)

{'score': 0.9867061972618103, 'start': 105, 'end': 109, 'answer': '1209'}
```

## Citing
BatteryBERT: A Pre-trained Language Model for Battery Database Enhancement


```shell
python run_ner.py --model_name_or_path batterydata/batterybert-cased --train_file tests/test_files/test_cner.json --validation_file tests/test_files/test_cner.json --text_column_name words --label_column_name labels --output_dir tests/cner_test_models --do_train --do_eval --max_seq_length 512  --pad_to_max_length true --overwrite_output_dir true --use_auth_token --max_train_samples 1 --max_eval_samples 1
```