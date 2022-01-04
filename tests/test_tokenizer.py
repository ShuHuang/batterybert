# -*- coding: utf-8 -*-
"""
batterybert.tests.test_tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test if a tokenizer is loaded correctly for pre-training and fine-tuning.
author: Shu Huang (sh2009@cam.ac.uk)
"""
import os
import unittest
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from batterybert.pretrain.tokenizer import PretrainTokenizer
from batterybert.finetune.tokenizer import QATokenizer



class TestPretrainTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        # Declaration
        tokenizer_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files/vocab.txt")
        pretrain_tokenizer = PretrainTokenizer(tokenizer_root=tokenizer_root)
        tokenizer = pretrain_tokenizer.get_tokenizer()

        # Assertion
        self.assertTrue(isinstance(tokenizer, PreTrainedTokenizer))


class TestQATokenizer(unittest.TestCase):
    def test_tokenizer(self):
        # Declaration
        tokenizer_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files/pretrain")
        qa_tokenizer = QATokenizer(tokenizer_root)
        tokenizer = qa_tokenizer.get_tokenizer()

        # Assertion
        self.assertTrue(isinstance(tokenizer, PreTrainedTokenizerFast))


if __name__ == '__main__':
    unittest.main()
