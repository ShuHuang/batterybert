# -*- coding: utf-8 -*-
"""
batterybert.pretrain.datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

prepare dataset
author: Shu Huang (sh2009@cam.ac.uk)
"""
from datasets import load_dataset
from .tokenizer import PretrainTokenizer


class PretrainDataset:
    """
    Dataset used for pre-training BatteryBERT.
    """

    def __init__(self, train_root, validation_root, tokenizer_root, cache_root=None, block_size=128):
        self.train_root = train_root
        self.validation_root = validation_root
        self.cache_root = cache_root
        self.tokenizer_root = tokenizer_root
        self.block_size = block_size

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def get_dataset(self):
        pretrain_datasets = load_dataset("text",
                                         data_files={"train": self.train_root, "validation": self.validation_root},
                                         cache_dir=self.cache_root)
        return pretrain_datasets

    def get_tokenized_datasets(self):
        pretrain_tokenizer = PretrainTokenizer(self.tokenizer_root, self.cache_root, do_lower_case=True)
        datasets = self.get_dataset()
        tokenized_datasets = datasets.map(pretrain_tokenizer.tokenize_function, batched=True, num_proc=4,
                                          remove_columns=["text"])
        lm_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            batch_size=1000,
            num_proc=4)
        return lm_datasets
