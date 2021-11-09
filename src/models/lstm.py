import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from data import ComicPanelBatch


class TextOnlyHeirarchicalLSTM(nn.Module):
    def __init__(self, vocab_len: int, word_embedding_dim: int = 256):
        super(TextOnlyHeirarchicalLSTM, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_len, embedding_dim=word_embedding_dim
        )
        self.lstm_box = nn.LSTM(
            input_size=word_embedding_dim,
            hidden_size=word_embedding_dim,
            batch_first=True,
        )
        self.lstm_panel = nn.LSTM(
            input_size=word_embedding_dim,
            hidden_size=word_embedding_dim,
            batch_first=True,
        )

    def forward(self, batch: ComicPanelBatch):
        batch_size, n_context, n_boxes_max, n_words_max = batch.context_words.shape

        # Generate embeddings for each speech box by summing the embeddings of the
        # individual words.
        context_box_embeddings = torch.sum(
            self.embedding(batch.context_words * batch.context_word_masks), dim=-2
        )
        answer_box_embeddings = torch.sum(
            self.embedding(batch.answer_words * batch.answer_word_masks), dim=-2
        )

        # Reshape tensors so that each panel is treated as an individual sequence of
        # speech boxes when feeding into LSTM.
        n_panels = batch_size * n_context
        num_boxes_per_panel = torch.sum(batch.context_box_masks, dim=-1).reshape(
            n_panels
        )
        context_box_embeddings = context_box_embeddings.reshape(
            (n_panels, n_boxes_max, -1)
        )

        # pack_padded_sequence will complain about sequences with length 0 (this
        # happens if there are no speech boxes in a panel). To get around this, we set
        # all the lengths of 0 to 1, but use a mask to zero out the panel embeddings.
        empty_panel_mask = (num_boxes_per_panel > 0).reshape(-1, 1)
        num_boxes_per_panel = num_boxes_per_panel + (num_boxes_per_panel == 0)

        context_box_embeddings_packed = pack_padded_sequence(
            input=context_box_embeddings,
            lengths=num_boxes_per_panel.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Use the final hidden state for the box-level LSTM as the panel embeddings.
        _, (panel_embeddings, _) = self.lstm_box(context_box_embeddings_packed)
        panel_embeddings = panel_embeddings.squeeze()

        # Zero out any panel embeddings that didn't contain speech boxes.
        panel_embeddings *= empty_panel_mask

        # Use the final hidden state of the panel-level LSTM as the embedding for the
        # entire context.
        panel_embeddings = panel_embeddings.reshape((batch_size, n_context, -1))
        _, (context_embeddings, _) = self.lstm_panel(panel_embeddings)
        context_embeddings = context_embeddings.reshape(batch_size, -1, 1)

        # Compute batched matrix product between each context vector and the
        # corresponding answer vectors.
        scores = torch.bmm(answer_box_embeddings, context_embeddings)
        scores = scores.reshape(batch_size, 3)
        return scores
