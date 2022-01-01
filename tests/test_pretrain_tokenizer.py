# -*- coding: utf-8 -*-
"""
batterybert.tests.test_pretrain_tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test if tokenizer is loaded correctly.
author: Shu Huang (sh2009@cam.ac.uk)
"""
import os
import unittest
from batterybert.pretrain.tokenizer import PretrainTokenizer


class TestTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        # Declaration
        tokenizer_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files/vocab.txt")
        pretrain_tokenizer = PretrainTokenizer(tokenizer_root=tokenizer_root)
        tokenizer = pretrain_tokenizer.get_tokenizer()

        # Testing
        vocab = tokenizer.name_or_path

        # Assertion
        self.assertEqual(tokenizer_root, vocab)


if __name__ == '__main__':
    unittest.main()
