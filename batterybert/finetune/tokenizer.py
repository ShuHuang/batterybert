# -*- coding: utf-8 -*-
"""
batterybert.finetune.tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare tokenizer
author: Shu Huang (sh2009@cam.ac.uk)
"""
from transformers import BertTokenizerFast


class FinetuneTokenizer:
    """
    Tokenizer for fine-tuning.
    """
    pass


class QATokenizer(FinetuneTokenizer):
    def __init__(self, model_root, cache_root=None, do_lower_case=True):
        """
        Required parameters for PretrainTokenizer
        :param tokenizer_root: pre-trained or default vocab file of tokenizer
        :param cache_root: cache directory of transformers
        :param do_lower_case: cased or uncased words for language model pre-training
        """
        self.tokenizer = model_root
        self.do_lower_case = do_lower_case
        self.cache_root = cache_root

    def get_tokenizer(self):
        """
        Get the tokenizer
        :return: Tokenizer
        """
        pretrain_tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer, do_lower_case=self.do_lower_case,
                                                           cache_dir=self.cache_root)
        return pretrain_tokenizer
