from typing import Dict, List

import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertModel,
    DistilBertTokenizer,
    DistilBertModel,
    ViTFeatureExtractor,
    ViTModel,
)

from data.text_cloze import TextClozeBatch
from data.visual_cloze import VisualClozeBatch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageTextTextClozeTransformerBaseline(nn.Module):
    def __init__(self, use_distilbert=False):
        super(ImageTextTextClozeTransformerBaseline, self).__init__()

        self.use_distilbert = use_distilbert
        if use_distilbert:
            self.bert_model = DistilBertModel.from_pretrained(
                'distilbert-base-uncased'
            ).to(device)
        else:
            self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

        self.vit_model = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k'
        ).to(device)

        self.lstm_panel = nn.LSTM(
            input_size=768,
            hidden_size=768,
            batch_first=True,
        )
        
        input_linear = 2 * 3 * 768 # 2 layers * 3 panels n_context * 768 hidden_size
        output_linear = 3 * 768 # 3 panels n_context * 768 hidden_size

        self.linear = nn.Linear(input_linear, output_linear).to(device)

    def forward(self, batch: VisualClozeBatch):
        batch_size = batch.batch_size
        n_context = batch.n_context

        # text embeddings
        panel_embeddings = self._get_bert_embeddings(batch.context_panel_bert_input)
        # print(panel_embeddings.shape)
        panel_embeddings = panel_embeddings.reshape(batch_size, n_context, -1)
        # print(panel_embeddings.shape)

        # image embeddings.
        panel_image_embeddings = self._get_visual_embeddings(
            batch.context_panel_vit_input
        )
        panel_image_embeddings = panel_image_embeddings.reshape(
            batch_size, n_context, -1
        )

        ### get combined image and text ###
        combined_image_text = torch.stack([panel_embeddings, panel_image_embeddings],axis=1)

        #need to flatten to get linear + put through
        input_to_linear = combined_image_text.reshape(batch_size, -1)
        out_linear = self.linear(input_to_linear)
        input_image_text = out_linear.reshape(batch_size, n_context, -1)

        #lstm with both image and text data:
        _, (_, context_embeddings) = self.lstm_panel(input_image_text)
        context_embeddings = context_embeddings.reshape(batch_size, -1, 1)

        # Answer embeddings.
        answer_embeddings = self._get_bert_embeddings(batch.answer_panel_bert_input)
        answer_embeddings = answer_embeddings.reshape(batch_size, 3, -1)

        # TODO: try cosine simlarity as well
        scores = torch.bmm(answer_panel_image_embeddings, context_embeddings)
        scores = scores.reshape(batch_size, 3)
        
        return scores

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

    def _get_visual_embeddings(self, vit_input: Dict):
        for key, tensor in vit_input.items():
            if isinstance(tensor, torch.Tensor):
                vit_input[key] = tensor.to(device, non_blocking=True)
        vit_outputs = self.vit_model(**vit_input)
        return vit_outputs.pooler_output
