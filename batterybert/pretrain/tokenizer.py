# -*- coding: utf-8 -*-
"""
batterybert.pretrain.tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

prepare dataset
author: Shu Huang (sh2009@cam.ac.uk)
"""
from transformers import BertTokenizer


class PretrainTokenizer:
    """
    Tokenizer for pre-training dataset.
    """

    def __init__(self, tokenizer_root, cache_root=None, do_lower_case=True):
        """
        Required parameters for PretrainTokenizer
        :param tokenizer_root: pre-trained or default vocab file of tokenizer
        :param cache_root: cache directory of transformers
        :param do_lower_case: cased or uncased words for language model pre-training
        """
        self.tokenizer = tokenizer_root
        self.do_lower_case = do_lower_case
        self.cache_root = cache_root

    def tokenize_function(self, examples):
        """
        Tokenize the Dataset object file
        """
        bert_tokenizer = self.get_tokenizer()
        return bert_tokenizer(examples["text"])

    def get_tokenizer(self):
        """
        Get the tokenizer
        :return: Tokenizer
        """
        pretrain_tokenizer = BertTokenizer.from_pretrained(self.tokenizer, do_lower_case=self.do_lower_case,
                                                           cache_dir=self.cache_root)
        return pretrain_tokenizer
