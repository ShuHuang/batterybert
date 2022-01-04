# -*- coding: utf-8 -*-
"""
batterybert.finetune.models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare fine-tuning models
author: Shu Huang (sh2009@cam.ac.uk)
"""
from transformers import BertConfig, BertForQuestionAnswering


class FinetuneModel:
    """

    """
    pass


class QAModel(FinetuneModel):
    """
    QA model of BatteryBERT.
    """
    def __init__(self, model_name_or_path, config=None):
        """
        Required parameters: config name or path.
        :param model_name_or_path: model name or path
        :param config: config name or path
        """
        self.model_name_or_path = model_name_or_path
        self.config = config

    def get_config(self):
        """

        :return:
        """
        config = BertConfig.from_pretrained(self.config if self.config else self.model_name_or_path)
        return config

    def get_model(self):
        """
        MaskedLM BERT model
        :return: BERTForQuestionAnswering class
        """
        model = BertForQuestionAnswering(self.get_config())
        return model
