from typing import Dict, List

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel

from data.text_cloze import TextClozeBatch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextOnlyTextClozeTransformerBaseline(nn.Module):
    def __init__(self, use_distilbert=False):
        super(TextOnlyTextClozeTransformerBaseline, self).__init__()

        self.use_distilbert = use_distilbert
        if use_distilbert:
            # self.bert_tokenizer = DistilBertTokenizer.from_pretrained(
            #     'distilbert-base-uncased'
            # )
            self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        else:
            # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

        self.lstm_panel = nn.LSTM(
            input_size=768,
            hidden_size=768,
            batch_first=True,
        )

    def forward(self, batch: TextClozeBatch):
        batch_size = batch.batch_size
        n_context = batch.n_context

        panel_embeddings = self._get_bert_embeddings(batch.context_panel_bert_input)
        panel_embeddings = panel_embeddings.reshape(batch_size, n_context, -1)

        # Answer embeddings.
        answer_embeddings = self._get_bert_embeddings(batch.answer_panel_bert_input)
        answer_embeddings = answer_embeddings.reshape(batch_size, 3, -1)

        _, (_, context_embeddings) = self.lstm_panel(panel_embeddings)
        context_embeddings = context_embeddings.reshape(batch_size, -1, 1)

        # TODO: try cosine simlarity as well
        scores = torch.bmm(answer_embeddings, context_embeddings)
        scores = scores.reshape(batch_size, 3)

        return scores, panel_embeddings, answer_embeddings

    def _get_bert_embeddings(self, bert_input: Dict):
        for key, tensor in bert_input.items():
            if isinstance(tensor, torch.Tensor):
                bert_input[key] = tensor.to(device, non_blocking=True)

        bert_outputs = self.bert_model(**bert_input)

        if self.use_distilbert:
            return bert_outputs[0][
                :, 0
            ]  # Distilbert returns a tuple where the 1st thing is hidden state.
        return bert_outputs.pooler_output
