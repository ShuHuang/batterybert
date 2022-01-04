# -*- coding: utf-8 -*-
"""
batterybert.extract.classify
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Classify battery or non-battery papers.

author: Shu Huang (sh2009@cam.ac.uk)
"""
from transformers import BertTokenizerFast, BertForSequenceClassification


class DocClassifier:
    """

    """
    def __init__(self, model_name_or_path):
        """

        :param model_name_or_path:
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        self.model = BertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)

    def classify(self, text, max_length=512):
        """

        :param text:
        :param max_length:
        :return:
        """
        tokenizer = self.tokenizer
        model = self.model
        inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        outputs = model(**inputs)
        probs = outputs[0].softmax(1)
        return probs.argmax().item()
