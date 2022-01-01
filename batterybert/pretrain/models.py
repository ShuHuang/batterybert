# -*- coding: utf-8 -*-
"""
batterybert.pretrain.models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare model
author: Shu Huang (sh2009@cam.ac.uk)
"""
from transformers import BertConfig, BertForMaskedLM


class PretrainModel:
    """
    MaskedLM model of BatteryBERT.
    """
    def __init__(self, config):
        """
        Required parameters: config name or path.
        :param config: config name or file path
        """
        self.checkpoint = config

    def get_config(self):
        config = BertConfig.from_pretrained(self.checkpoint)
        return config

    def get_model(self):
        """
        MaskedLM BERT model
        :return: BERTForMaskedLM
        """
        model = BertForMaskedLM(self.get_config())
        return model
