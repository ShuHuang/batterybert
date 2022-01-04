# -*- coding: utf-8 -*-
"""
batterybert.tests.test_doc_classify
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test binary battery text classifier
author: Shu Huang (sh2009@cam.ac.uk)
"""
import unittest
from batterybert.apps import DocClassifier


class TestDocClassifier(unittest.TestCase):
    def test_doc_classifier_true(self):
        # Declaration
        model_name = "batterydata/test4"
        battery_text = "A database of battery materials is presented which comprises a total of 292,313 data " \
                       "records, with 214,617 unique chemical-property data relations between 17,354 unique " \
                       "chemicals and up to five material properties: capacity, voltage, conductivity, " \
                       "Coulombic efficiency and energy. 117,403 data are multivariate on a property where it " \
                       "is the dependent variable in part of a data series. The database was auto-generated " \
                       "by mining text from 229,061 academic papers using the chemistry-aware natural " \
                       "language processing toolkit, ChemDataExtractor version 1.5, which was modified for " \
                       "the specific domain of batteries. The collected data can be used as a " \
                       "representative overview of battery material information that is contained within text of " \
                       "scientific papers. Public availability of these data will also enable battery materials " \
                       "design and prediction via data-science methods. To the best of our knowledge, this is the " \
                       "first auto-generated database of battery materials extracted from a relatively large number " \
                       "of scientific papers. We also provide a Graphical User Interface (GUI) to aid the use of " \
                       "this database."
        classifier = DocClassifier(model_name)

        # Testing. Expect: 1 (battery text)
        category = classifier.classify(battery_text)

        # Assertion
        self.assertEqual(category, 1)

    def test_doc_classifier_false(self):
        # Declaration
        model_name = "batterydata/test4"
        non_battery_text = "The emergence of “big data” initiatives has led to the need for tools that can " \
                           "automatically apps valuable chemical information from large volumes of unstructured " \
                           "data, such as the scientific literature. Since chemical information can be present in " \
                           "figures, tables, and textual paragraphs, successful information extraction often " \
                           "depends on the ability to interpret all of these domains simultaneously. We present a " \
                           "complete toolkit for the automated extraction of chemical entities and their associated " \
                           "properties, measurements, and relationships from scientific documents that can be used " \
                           "to populate structured chemical databases. Our system provides an extensible, chemistry-" \
                           "aware, natural language processing pipeline for tokenization, part-of-speech tagging, " \
                           "named entity recognition, and phrase parsing."
        classifier = DocClassifier(model_name)

        # Testing. Expect: 0 (non-battery text)
        category = classifier.classify(non_battery_text)

        # Assertion
        self.assertEqual(category, 0)


if __name__ == '__main__':
    unittest.main()
