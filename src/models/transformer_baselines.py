from typing import List

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel

from data import ComicPanelBatch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextOnlyTransformerBaseline(nn.Module):
    def __init__(self, idx_to_word, use_distilbert=False):
        super(TextOnlyTransformerBaseline, self).__init__()

        self.idx_to_word = idx_to_word

        self.use_distilbert = use_distilbert
        if use_distilbert:
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased'
            )
            self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

        self.lstm_panel = nn.LSTM(input_size=768, hidden_size=768, batch_first=True,)

        # # Freeze BERT parameters.
        # for param in self.bert_model.parameters():
        #     param.requires_grad = False

    def forward(self, batch: ComicPanelBatch):
        # print(f'{batch.context_box_text[0] = }')
        # print(f'{batch.answer_text[0] = }')
        # print(f'{batch.answer_words[0] = }')
        # print(f'{batch.answer_word_masks[0] = }')
        # assert False

        batch_size, n_context, n_boxes_max, n_words_max = batch.context_words.shape

        # Join all 3 of the text boxes in a panel into a single sentence.
        context_box_text_joined = [
            ' '.join(batch.context_box_text[i : i + 3])
            for i in range(0, len(batch.context_box_text), 3)
        ]

        panel_embeddings = self._get_bert_embeddings(context_box_text_joined)
        panel_embeddings = panel_embeddings.reshape(batch_size, n_context, -1)

        # Answer embeddings.
        answer_embeddings = self._get_bert_embeddings(batch.answer_text)
        answer_embeddings = answer_embeddings.reshape(batch_size, 3, -1)

        _, (_, context_embeddings) = self.lstm_panel(panel_embeddings)
        context_embeddings = context_embeddings.reshape(batch_size, -1, 1)

        # TODO: try cosine simlarity as well
        scores = torch.bmm(answer_embeddings, context_embeddings)
        scores = scores.reshape(batch_size, 3)

        return scores

    def _get_bert_embeddings(self, sentences: List[str]):
        bert_input = self.bert_tokenizer(
            sentences, return_tensors='pt', padding=True, truncation=True
        )
        for key, tensor in bert_input.items():
            if isinstance(tensor, torch.Tensor):
                bert_input[key] = tensor.to(device)

        bert_outputs = self.bert_model(**bert_input)

        if self.use_distilbert:
            return bert_outputs[0][
                :, 0
            ]  # Distilbert returns a tuple where the 1st thing is hidden state.
        return bert_outputs.pooler_output
