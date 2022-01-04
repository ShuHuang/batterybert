# -*- coding: utf-8 -*-
"""
batterybert.finetune.dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare fine-tuning dataset
coauthor: Shu Huang (sh2009@cam.ac.uk)
coauthor: The HuggingFace Team
"""
import torch
import pandas as pd
from datasets import load_dataset
from .tokenizer import FinetuneTokenizerFast


class FinetuneDataset(torch.utils.data.Dataset):
    pass


class DocDataset(FinetuneDataset):
    """
    Document dataset
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


class PaperDataset(FinetuneDataset):
    """
    Document dataset
    """
    def __init__(self, model_root, training_root, eval_root):
        """

        :param model_root: pre-trained model
        :param training_root: training set of the paper corpus.
               csv format (columns: doi,date,year,title,journal,abstract,has_full_text,label).
        :param eval_root: validation set of the paper corpus. Same format as training set.
        """
        self.model_root = model_root
        self.training_root = training_root
        self.test_root = eval_root

    def get_dataset(self, max_length=512):
        """
        Get training data and eval data of the paper corpus.
        :param max_length: max length of the tokenized text
        :return: DocDataset of training data and eval data
        """
        tokenizer = FinetuneTokenizerFast(self.model_root).get_tokenizer()
        training_data = pd.read_csv(self.training_root)
        test_data = pd.read_csv(self.test_root)

        if "abstract" not in training_data or "abstract" not in test_data:
            raise ValueError("Need a valid dataset including abstract.")

        training_data['abstract'] = training_data['abstract'].astype(str)
        train_text = training_data['abstract'].values.tolist()
        train_label = training_data['label'].to_numpy()

        test_data['abstract'] = test_data['abstract'].astype(str)
        test_text = test_data['abstract'].values.tolist()
        test_label = test_data['label'].to_numpy()

        train_encodings = tokenizer(train_text, truncation=True, padding=True, max_length=max_length)
        valid_encodings = tokenizer(test_text, truncation=True, padding=True, max_length=max_length)

        # convert our tokenized data into a torch Dataset
        train_dataset = DocDataset(train_encodings, train_label)
        eval_dataset = DocDataset(valid_encodings, test_label)

        return train_dataset, eval_dataset


class QADataset(FinetuneDataset):
    """
    Dataset used for fine-tuning question answering of BatteryBERT.
    """

    def __init__(self, model_root, dataset_name=None, train_root=None, validation_root=None, cache_root=None):
        """
        Required parameters for PretrainDataset
        :param dataset_name: QA dataset name in default datasets model
        :param train_root: training text file for BatteryBERT pre-training
        :param validation_root: test text file for BatteryBERT pre-training
        :param model_root: pre-trained model
        :param cache_root: cache directory of transformers
        """
        self.dataset_name = dataset_name
        self.train_root = train_root
        self.validation_root = validation_root
        self.cache_root = cache_root
        self.model_root = model_root

    def get_dataset(self, dataset_config_name=None):
        """
        Load the pretrained file from .txt into Dataset
        :return: Dataset
        """
        if self.dataset_name is not None:
            raw_datasets = load_dataset(self.dataset_name, dataset_config_name, cache_dir=self.cache_root)
        elif self.dataset_name is None and self.train_root is None and self.validation_root is None:
            raw_datasets = load_dataset('squad', dataset_config_name, cache_dir=self.cache_root)
        else:
            data_files = {}
            if self.train_root is not None:
                data_files["train"] = self.train_root
                extension = self.train_root.split(".")[-1]

            if self.validation_root is not None:
                data_files["validation"] = self.validation_root
                extension = self.validation_root.split(".")[-1]
            raw_datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir=self.cache_root)

        return raw_datasets

    def get_train_dataset(self, dataset_config_name=None):
        """

        :param dataset_config_name:
        :return:
        """
        raw_datasets = self.get_dataset(dataset_config_name)
        column_names = raw_datasets["train"].column_names
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(
            self.prepare_train_features,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on train dataset",
        )
        return train_dataset

    def get_eval_dataset(self, dataset_config_name=None):
        """

        :param dataset_config_name:
        :return:
        """
        raw_datasets = self.get_dataset(dataset_config_name)
        column_names = raw_datasets["validation"].column_names
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")

        eval_dataset = raw_datasets["validation"]
        eval_dataset = eval_dataset.map(
            self.prepare_validation_features,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on validation dataset",
        )
        return eval_dataset

    # Training preprocessing
    def prepare_train_features(self, examples, max_seq_length=384, stride=128, pad_to_max_length=True):
        """
        Prepare train features.
        author: The HuggingFace team
        :param examples:
        :param max_seq_length:
        :param stride:
        :param pad_to_max_length:
        :return:
        """
        # Remove the left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenizer = FinetuneTokenizerFast(self.model_root).get_tokenizer()
        pad_on_right = tokenizer.padding_side == "right"
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # Validation preprocessing
    def prepare_validation_features(self, examples, max_seq_length=384, stride=128, pad_to_max_length=True):
        """
        author: The HuggingFace team
        :param examples:
        :param max_seq_length:
        :param stride:
        :param pad_to_max_length:
        :return:
        """
        # Remove the left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        tokenizer = FinetuneTokenizerFast(self.model_root).get_tokenizer()
        pad_on_right = tokenizer.padding_side == "right"
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
