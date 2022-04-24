# -*- coding: utf-8 -*-
"""
batterybert.finetune.bertcrf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conditional random field model for BERT CNER.
author: Shu Huang <sh2009@cam.ac.uk>
"""
from typing import Optional, Tuple, Union
from transformers import BertPreTrainedModel, BertModel, add_start_docstrings
from transformers.file_utils import add_start_docstrings_to_model_forward
from transformers.models.bert.modeling_bert import BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING
import torch
from torch import nn
from torchcrf import CRF
from .utils import CNEROutput


@add_start_docstrings(
    """
    Bert Model with a token classification head and a condition random field layer on top (a linear layer on top of
     the hidden-states output) e.g. for amed-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertCrfForTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CNEROutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Final hidden layer
        sequence_output = outputs[0]
        # Dropout on valid hidden layer output
        sequence_output = self.dropout(sequence_output)
        # Classification logits
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss, tags = -self.crf(logits, labels, mask=attention_mask), self.crf.decode(logits, mask=attention_mask)
        else:
            tags = self.crf.decode(logits)
        # tags = torch.Tensor(tags)

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CNEROutput(
            loss=loss,
            logits=logits,
            # tags=tags,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a token classification head, a LSTM layer and a condition random field layer on top (a linear 
    layer on top of the hidden-states output) e.g. for amed-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertLstmCrfForTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, dropout=config.hidden_dropout_prob,
                              batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CNEROutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Final hidden layer
        sequence_output = outputs[0]
        # Dropout on valid hidden layer output
        sequence_output = self.dropout(sequence_output)
        # LSTM Output
        lstm_output, hc = self.bilstm(sequence_output)
        # Classification logits
        logits = self.classifier(lstm_output)

        loss = None
        if labels is not None:
            loss, tags = -self.crf(logits, labels, mask=attention_mask), self.crf.decode(logits, mask=attention_mask)
        else:
            tags = self.crf.decode(logits)
        # tags = torch.Tensor(tags)

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CNEROutput(
            loss=loss,
            logits=logits,
            # tags=tags,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
