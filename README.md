# BatteryBERT
BatteryBERT Pre-training & Fine-tuning: Question Answering, Document Classification, and Data Extraction.

## Features

- BERT pre-training
- Fine-tuning: question answering and document classification

## Installation

## Example Usage
Run pre-training from checkpoints:
```
python run_pretrain.py --train_root 'tests/test_files/test_text_example.txt' --eval_root 'tests/test_files/test_text_example.txt' --output_dir 'tests/test_files/pretrain/' --tokenizer_root 'tests
/test_files/vocab.txt' --checkpoint='tests/test_files/pretrain'
```

## Citing