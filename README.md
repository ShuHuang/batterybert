# BatteryBERT
BatteryBERT Pre-training & Fine-tuning: Question Answering, Document Classification, and Data Extraction.

## Features

- BERT pre-training
- Fine-tuning: question answering and document classification
- Use the fine-tuned model
- Data extraction and database enhancement

## Installation

## Example Usage
### Run pre-training
```
python run_pretrain.py --train_root 'tests/test_files/test_text_example.txt' --eval_root 'tests/test_files/test_text_example.txt' --output_dir 'tests/test_files/pretrain/' --tokenizer_root 'tests
/test_files/vocab.txt' --checkpoint='tests/test_files/pretrain'
```

###Run fine-tuning
Run question answering:
```
 python .\run_finetune_qa.py --model_name_or_path .\tests\test_files\pretrain\ --output_dir .\tests\test_files\qa\ --do_train True --do_eval False
```
## Citing
BatteryBERT: A Pre-trained Language Model for Battery Database Enhancement