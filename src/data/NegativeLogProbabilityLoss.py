import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def shift_up(labels):
    labels = labels + 1
    labels[labels > 2] = 0
    return labels


def auxiliary_loss(labels, context_emb, answer_emb, C = 1):
    #annealing factor, needs to be < 10 otherwise get np.inf due to mixed precision

    #batch size
    bs = labels.shape[0]
    indexer = torch.arange(bs)
    
    #get incorrect batches of labels 
    incorrect_labels1 = shift_up(labels)
    incorrect_labels2 = shift_up(incorrect_labels1)

    #get answer_embeddings which are correct and incorrect for each batch
    correct_answers = answer_emb[indexer, labels].unsqueeze(1).permute(0,2,1)
    inccorrect_answers1 = answer_emb[indexer, incorrect_labels1].unsqueeze(1).permute(0,2,1)
    inccorrect_answers2 = answer_emb[indexer, incorrect_labels2].unsqueeze(1).permute(0,2,1)

    #get dot product (similarity measure)
    correct = batch_cosine_similarity(context_emb, correct_answers, bs)
    
    incorrect1 = batch_cosine_similarity(context_emb, inccorrect_answers1, bs)
    incorrect2 = batch_cosine_similarity(context_emb, inccorrect_answers1, bs)

    #term to maximize:
    max_term = - C * (incorrect1 + incorrect1).sum()

    #term to minimize:
    pre_log_min = (torch.e ** (C * correct)).sum(axis=1) #sum along (v_i, v_j) pairs within sample 
    min_term = torch.log(pre_log_min).sum() #sum along N (batch size)

    #aux loss:
    aux_loss = 1/9 * (max_term + min_term)
    
    return aux_loss

def batch_cosine_similarity(context_emb, answer_emb, bs):
    #they have batch matrix product but not cosine similarity...
    correct = torch.bmm(context_emb, answer_emb).reshape(bs, -1)
    
    #get magnitude
    correct_norm = torch.linalg.norm(context_emb).item() * torch.linalg.norm(answer_emb).item()
    correct = correct/correct_norm
    return correct

class NegativeLogProbabilityLoss(_Loss):
    def __init__(self, C=0.01, weight=0.5):
        super(NegativeLogProbabilityLoss, self).__init__()
        self.C = C
        self.weight = weight

    def forward(self, logits, labels, context_emb, answer_emb):
        
        #losses
        primary_loss = F.cross_entropy(logits, labels)
        
        print(primary_loss)
        aux_loss = auxiliary_loss(labels, context_emb, answer_emb, self.C)
        print(aux_loss)
        
        #get the weights of the losses
        return primary_loss * (self.weight) + aux_loss * (1 - self.weight)
        
        