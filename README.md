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

Use the device data extractor:
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

[{'type': 'anode', 'answer': 'grapite', 'score': 0.9736555218696594, 'context': 'The anode of this battery is grapite.'}]
```

Use the general Q&A agent:
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