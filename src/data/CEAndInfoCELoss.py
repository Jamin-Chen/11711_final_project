from torch.nn.modules.loss import _Loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_similarities(
    answer_emb, context_emb, labels, C, batch_size, device='cuda'
):
    n_context = 3
    n_total_comparisons = 9

    # cosine similarities
    cosine_sim = torch.nn.CosineSimilarity(dim=3)
    # TODO: Is this the right order to unsqueeze dimensions? Think about this more.
    similarities = cosine_sim(answer_emb.unsqueeze(2), context_emb.unsqueeze(1))

    # masks
    correct = torch.zeros(batch_size, n_context).to(device)
    correct[np.arange(len(correct)), labels] = 1

    # For some reason, doing incorrect = correct == False leads to NaN in backwards.
    incorrect = torch.zeros_like(correct).to(device)
    incorrect[np.arange(len(correct)), ~labels] = 1

    # calculate terms
    max_values = -1 * ((C * similarities).sum(axis=-1) * correct).sum(axis=-1).sum()

    # Flatten dimensions 1 and 2 together before applying logsumexp.
    min_values = torch.logsumexp(
        (C * similarities * incorrect.detach().unsqueeze(2)).view(batch_size, -1), axis=-1
    )
    # Sum over examples in batch.
    min_values = min_values.sum()

    # combine terms
    final_loss = (
        (max_values + min_values) * (1 / n_total_comparisons) * (1 / batch_size)
    )

    return final_loss


class CEAndInfoCELoss(nn.Module):
    def __init__(self, lambda_=0.5, device='cuda'):
        super(CEAndInfoCELoss, self).__init__()

        # annealing C values
        self.n_batches = 0
        self.ramp_up = 10000
        self.sqrt_d = np.sqrt(768)

        self.lambda_ = lambda_
        self.device = device

    def forward(self, logits, labels, context_emb, answer_emb, train=True):
        batch_size = labels.shape[0]

        # calculate C:
        # for every batch up to 10K training batches, will get closer and closer to sqrt d from 0 (until it is perpetually at sqrt(d).
        loss_c = min(self.n_batches / self.ramp_up, 1) * self.sqrt_d

        # losses
        ce_loss = F.cross_entropy(logits, labels)

        info_ce_loss = calculate_similarities(
            answer_emb, context_emb, labels, loss_c, batch_size, self.device
        )

        if train:
            # update counter
            self.n_batches += 1

        # get the weights of the losses
        return ce_loss * (self.lambda_) + info_ce_loss * (1 - self.lambda_)
