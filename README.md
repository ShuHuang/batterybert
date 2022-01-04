# BatteryBERT
BatteryBERT Pre-training & Fine-tuning: Question Answering, Document Classification, and Data Extraction.

## Features

- BERT pre-training
- Fine-tuning: question answering and document classification
- Use the fine-tuned model
- Data extraction and database enhancement

## Installation

## Example Usage
### Pre-train BatteryBERT
TODO: change the path for all the command
```
python run_pretrain.py --train_root 'tests/test_files/test_text_example.txt' --eval_root 'tests/test_files/test_text_example.txt' --output_dir 'tests/test_files/pretrain/' --tokenizer_root 'tests/test_files/vocab.txt' --checkpoint='tests/test_files/pretrain'
```

### Fine-tune BatteryBERT 
Run fine-tuning: question answering:
```
python .\run_finetune_qa.py --model_name_or_path .\tests\test_files\pretrain\ --output_dir .\tests\test_files\qa\ --do_train True --do_eval False
```

Run fine-tuning: document classification
```
python .\run_finetune_doc_classify.py --model_name_or_path .\tests\test_files\pretrain\ --output_dir .\tests\test_files\doc\ --train_root .\tests\test_files\doc\training_data.csv --eval_root .\tests\test_files\doc\test_data.csv
```
### Use the document classifier
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

## Citing
BatteryBERT: A Pre-trained Language Model for Battery Database Enhancement