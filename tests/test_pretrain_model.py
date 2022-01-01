# -*- coding: utf-8 -*-
"""
batterybert.tests.test_pretrain_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test if model is loaded correctly.
author: Shu Huang (sh2009@cam.ac.uk)
"""
import unittest
from batterybert.pretrain.models import PretrainModel


class TestBertModel(unittest.TestCase):
    def test_bertmodel(self):
        # Declaration
        model_name = 'bert-base-cased'
        pretrain_model = PretrainModel(model_name)
        model = pretrain_model.get_model()

        # Testing
        model_name = model._get_name()

        # Assertion
        self.assertEqual('BertForMaskedLM', model_name)


if __name__ == '__main__':
    unittest.main()
