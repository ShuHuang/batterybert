# -*- coding: utf-8 -*-
"""
batterybert.tests.test_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test if a model is loaded correctly.
author: Shu Huang (sh2009@cam.ac.uk)
"""
import unittest
from batterybert.pretrain.models import PretrainModel
from batterybert.finetune.models import QAModel


class TestLMModel(unittest.TestCase):
    def test_bert_model(self):
        # Declaration
        model_name = 'bert-base-cased'
        pretrain_model = PretrainModel(model_name)
        model = pretrain_model.get_model()

        # Testing
        model_name = model._get_name()

        # Assertion
        self.assertEqual('BertForMaskedLM', model_name)


class TestQAModel(unittest.TestCase):
    def test_qa_model(self):
        # Declaration
        model_name = 'batterydata/test1'
        pretrain_model = QAModel(model_name)
        model = pretrain_model.get_model()

        # Testing
        model_name = model._get_name()

        # Assertion
        self.assertEqual('BertForQuestionAnswering', model_name)


if __name__ == '__main__':
    unittest.main()
