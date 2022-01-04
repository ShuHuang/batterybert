# -*- coding: utf-8 -*-
"""
batterybert.tests.test_device_extract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test the device data extraction.
author: Shu Huang (sh2009@cam.ac.uk)
"""
import unittest
from batterybert.apps import DeviceDataExtractor


class TestDeviceDataExtractor(unittest.TestCase):
    def test_anode_extract(self):
        # Declaration
        model_name = "batterydata/test1"
        text = "The anode of this Li-ion battery is graphite."
        extractor = DeviceDataExtractor(model_name)
        expected_answer = 'graphite'

        # Testing
        results = extractor.extract(text)[0]['answer']

        # Assertion
        self.assertEqual(expected_answer, results)

    def test_electrolyte_extract(self):
        # Declaration
        model_name = "batterydata/test1"
        text = "The typical non-aqueous electrolyte for commercial Li-ion cells is a solution of LiPF6 in linear " \
               "and cyclic carbonates such as dimethyl carbonate and ethylene carbonate, respectively [1], [2]."
        extractor = DeviceDataExtractor(model_name)
        expected_answer = 'a solution of LiPF6 in linear and cyclic carbonates'

        # Testing
        results = extractor.extract(text)[0]['answer']

        # Assertion
        self.assertEqual(expected_answer, results)


if __name__ == '__main__':
    unittest.main()

