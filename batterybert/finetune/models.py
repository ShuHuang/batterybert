# -*- coding: utf-8 -*-
"""
batterybert.finetune.models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare fine-tuning models
author: Shu Huang (sh2009@cam.ac.uk)
"""
from transformers import BertConfig, BertForQuestionAnswering, BertForSequenceClassification


class FinetuneModel:
    """
    Models for fine-tuning.
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
        Get BERTConfig
        :return: BERTConfig
        """
        config = BertConfig.from_pretrained(self.config if self.config else self.model_name_or_path)
        return config


class QAModel(FinetuneModel):
    """
    QA model of BatteryBERT.
    """
    def get_model(self):
        """
        BERTForQuestionAnswering model
        :return: BERTForQuestionAnswering class
        """
        model = BertForQuestionAnswering(self.get_config())
        return model


class DocClassModel(FinetuneModel):
    """
    Document classification model of BatteryBERT.
    """
    def get_model(self):
        """
        BertForSequenceClassification
        :return: BertForSequenceClassification class
        """
        model = BertForSequenceClassification(self.get_config())
        return model
