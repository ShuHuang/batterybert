# -*- coding: utf-8 -*-
"""
batterybert.extract.extract_device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extract the device material from the battery text.

author: Shu Huang (sh2009@cam.ac.uk)
"""
from transformers.pipelines import pipeline


class DeviceDataExtractor:
    """
    Device data extractor
    """
    def __init__(self, model_name_or_path):
        """

        :param model_name_or_path: the fine-tuned model
        """
        self.model = pipeline('question-answering', model=model_name_or_path, tokenizer=model_name_or_path)

    def extract(self, context, threshold=0, num_answer=3):
        """

        :param context:
        :param threshold:
        :param num_answer:
        :return:
        """
        answers = []
        device_words = ['anode', 'cathode', 'electrolyte']
        for word in device_words:
            words = word + 's'
            cased_word = word.capitalize()
            cased_words = words.capitalize()
            if word in context or words in context or cased_word in context or cased_words in context:
                qa_input = {'question': "What is the {}?".format(word), 'context': context}
                res = self.model(qa_input, top_k=num_answer)
                for answer in res:
                    # Manual filters for answers
                    if 'battery' in answer['answer'] or 'batteries' in answer['answer']:
                        continue
                    if word == 'electrolyte' and 'lithium' in answer['answer']:
                        continue
                    # ChemDataExtractor is not included in here for simplification.
                    # cem_doc = Document(answer['answer'])
                    # print(cem_doc)
                    if answer['score'] > threshold:  # cem_doc.cems != [] and
                        this_answer = {"type": word, "answer": answer['answer'],
                                       "score": answer['score'], "context": context}
                        answers.append(this_answer)
                        break
        return answers
