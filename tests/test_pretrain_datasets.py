# -*- coding: utf-8 -*-
"""
batterybert.tests.test_pretrain_datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test if dataset is loaded correctly.
author: Shu Huang (sh2009@cam.ac.uk)
"""
import os
import unittest
from batterybert.pretrain.datasets import PretrainDataset


class TestDataset(unittest.TestCase):
    def test_dataset(self):
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


if __name__ == '__main__':
    unittest.main()
