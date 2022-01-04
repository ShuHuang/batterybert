# -*- coding: utf-8 -*-
"""
batterybert.apps.classify
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Classify battery or non-battery text.

author: Shu Huang (sh2009@cam.ac.uk)
"""
from transformers import BertTokenizerFast, BertForSequenceClassification


class DocClassifier:
    """
    Binary classifier for battery or non-battery text.
    """
    def __init__(self, model_name_or_path):
        """

        :param model_name_or_path: the fine-tuned model
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        self.model = BertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)

    def classify(self, text, max_length=512):
        """
        :param text: the text to be classified
        :param max_length: max length that BatteryBERT can handle
        :return: category[int]. 0 for non-battery text, 1 for battery text.
        """
        tokenizer = self.tokenizer
        model = self.model
        inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        outputs = model(**inputs)
        probs = outputs[0].softmax(1)
        return probs.argmax().item()
