# -*- coding: utf-8 -*-
"""
batterybert.pretrain.datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare dataset
author: Shu Huang (sh2009@cam.ac.uk)
"""
from datasets import load_dataset
from .tokenizer import PretrainTokenizer


class PretrainDataset:
    """
    Dataset used for pre-training BatteryBERT.
    """

    def __init__(self, train_root, validation_root, tokenizer_root, cache_root=None):
        """
        Required parameters for PretrainDataset
        :param train_root: training text file for BatteryBERT pre-training
        :param validation_root: test text file for BatteryBERT pre-training
        :param tokenizer_root: pre-trained or default vocab file of tokenizer
        :param cache_root: cache directory of transformers
        """
        self.train_root = train_root
        self.validation_root = validation_root
        self.cache_root = cache_root
        self.tokenizer_root = tokenizer_root

    @staticmethod
    def group_texts(examples, block_size=128):
        """
        Help function for group texts
        :param examples: Dataset object of example files
        :param block_size: block size for the text
        :return: grouped text as Dataset
        """
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def get_dataset(self):
        """
        Load the pretrained file from .txt into Dataset
        :return: Dataset
        """
        pretrain_datasets = load_dataset("text",
                                         data_files={"train": self.train_root, "validation": self.validation_root},
                                         cache_dir=self.cache_root)
        return pretrain_datasets

    def get_tokenized_datasets(self, batched=True, batch_size=1000, num_proc=4):
        """
        Get tokenized language model datasets
        :return: Tokenized Dataset of grouped text
        """
        pretrain_tokenizer = PretrainTokenizer(self.tokenizer_root, self.cache_root, do_lower_case=True)
        dataset = self.get_dataset()
        tokenized_datasets = dataset.map(pretrain_tokenizer.tokenize_function, batched=True, num_proc=4,
                                         remove_columns=["text"])
        lm_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=batched,
            batch_size=batch_size,
            num_proc=num_proc)
        return lm_datasets
