# -*- coding: utf-8 -*-
"""
batterybert.tests.test_qa_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test the Q&A agent
author: Shu Huang (sh2009@cam.ac.uk)
"""
import unittest
from batterybert.apps import QAAgent


class TestQAAgent(unittest.TestCase):
    def test_qa_agent(self):
        # Declaration
        model_name = "batterydata/test1"
        question = "When was University of Cambridge founded?"
        context = "The University of Cambridge is a collegiate research university in Cambridge, United Kingdom. " \
                  "Founded in 1209 and granted a royal charter by Henry III in 1231, Cambridge is the second-oldest " \
                  "university in the English-speaking world and the world's fourth-oldest surviving university."

        qa_agent = QAAgent(model_name)
        expected_answer = '1209'

        # Testing
        results = qa_agent.answer(question=question, context=context)['answer']

        # Assertion
        self.assertEqual(expected_answer, results)


if __name__ == '__main__':
    unittest.main()

