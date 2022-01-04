# BatteryBERT
BatteryBERT Pre-training & Fine-tuning: Question Answering, Document Classification, and Data Extraction.

## Features

- BERT pre-training
- Fine-tuning: question answering and document classification
- Use the fine-tuned model
- Data extraction and database enhancement

## Installation

## Example Usage
### BatteryBERT Pre-training
TODO: change the path for all the command
```
python run_pretrain.py --train_root 'tests/test_files/test_text_example.txt' --eval_root 'tests/test_files/test_text_example.txt' --output_dir 'tests/test_files/pretrain/' --tokenizer_root 'tests/test_files/vocab.txt' --checkpoint='tests/test_files/pretrain'
```

### BatteryBERT Fine-tuning
Run fine-tuning: question answering
```
python .\run_finetune_qa.py --model_name_or_path .\tests\test_files\pretrain\ --output_dir .\tests\test_files\qa\ --do_train True --do_eval False
```

Run fine-tuning: document classification
```
python .\run_finetune_doc_classify.py --model_name_or_path .\tests\test_files\pretrain\ --output_dir .\tests\test_files\doc\ --train_root .\tests\test_files\doc\training_data.csv --eval_root .\tests\test_files\doc\test_data.csv
```
### Use the fine-tuned BatteryBERT
Use the battery paper classifier:
```python
>>> from batterybert.extract import DocClassifier
>>> # Model name to be changed after published
>>> model_name = "batterydata/test4"
>>> sample_text = "sample text"
>>> classifier = DocClassifier(model_name)
>>> # 0 for non-battery text, 1 for battery text
>>> category = classifier.classify(sample_text)
>>> print(category)
=======================================================================================
0
```

Use the device data extractor:
```python
>>> from batterybert.extract import DeviceDataExtractor
>>> # Model name to be changed after published
>>> model_name = "batterydata/test1"
>>> sample_text = "The anode of this Li-ion battery is graphite."
>>> extractor = DeviceDataExtractor(model_name)
>>> # Set the confidence score threshold
>>> result = extractor.extract(sample_text, threshold=0.1)
>>> print(result)
=======================================================================================
[{'type': 'anode', 'answer': 'grapite', 'score': 0.9736555218696594, 'context': 'The anode of this battery is grapite.'}]
```

## Citing
BatteryBERT: A Pre-trained Language Model for Battery Database Enhancement