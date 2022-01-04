# -*- coding: utf-8 -*-
"""
batterybert.tests.test_dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test if a dataset is loaded correctly.
author: Shu Huang (sh2009@cam.ac.uk)
"""
import os
import unittest
import torch
from datasets import Dataset
from batterybert.pretrain.dataset import PretrainDataset
from batterybert.finetune.dataset import QADataset, PaperDataset


class TestPretrainDataset(unittest.TestCase):
    def test_pretrain_dataset(self):
        # Declaration
        train_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files/training_text_example.txt")
        test_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files/test_text_example.txt")
        tokenizer_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files/vocab.txt")
        dataset = PretrainDataset(train_root, test_root, tokenizer_root)
        results = dataset.get_tokenized_datasets()

        # Testing
        text_keys = list(results.keys())

        # Assertion
        self.assertEqual(['train', 'validation'], text_keys)


class TestQADataset(unittest.TestCase):
    def test_qa_train_dataset(self):
        # Declaration
        tokenizer_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files/pretrain")
        dataset = QADataset(tokenizer_root)
        train_dataset = dataset.get_train_dataset()

        # Assertion
        self.assertEqual(isinstance(train_dataset, Dataset), True)

    def test_qa_eval_dataset(self):
        # Declaration
        tokenizer_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files/pretrain")
        dataset = QADataset(tokenizer_root)
        eval_dataset = dataset.get_eval_dataset()

        # Assertion
        self.assertEqual(isinstance(eval_dataset, Dataset), True)


class TestPaperDataset(unittest.TestCase):
    def test_paper_dataset(self):
        # Declaration
        tokenizer_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files/pretrain")
        dataset = PaperDataset(tokenizer_root, training_root=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                          "test_files/doc/training_data.csv"),
                               eval_root=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "test_files/doc/test_data.csv"))
        train_dataset, eval_dataset = dataset.get_dataset()

        # Assertion
        self.assertEqual(isinstance(train_dataset, torch.utils.data.Dataset), True)


if __name__ == '__main__':
    unittest.main()
