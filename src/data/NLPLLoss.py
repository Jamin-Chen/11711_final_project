from torch.nn.modules.loss import _Loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_similarities(answer_emb, context_emb, labels, C, bs, device="cuda"):
    n_context = 3
    n_total_comparisons = 9
    
    #cosine similarities
    cosine_sim = torch.nn.CosineSimilarity(dim=3)
    similarities = cosine_sim(answer_emb.unsqueeze(1), context_emb.unsqueeze(2))

    #masks
    correct = torch.zeros(bs, n_context).to(device).bool()
    correct[np.arange(len(correct)),labels] = 1
    incorrect = correct == False

    #calculate terms
    max_values = -1 * ((C * similarities).sum(axis=-1) * correct).sum(axis=-1).sum()
    min_values = torch.log(((torch.e ** (C* similarities)).sum(axis=-1) * incorrect).sum(axis=1)).sum()

    #combine terms
    final_loss = (max_values + min_values) * (1/n_total_comparisons) * (1/bs)
    
    return final_loss

#get cosine similarities

class NLPLLoss(nn.Module):
    def __init__(self, C=0.1, weight=0.5, device="cuda"):
        super(NLPLLoss, self).__init__()
        
        self.C = C
        self.weight = weight
        self.device = device

    def forward(self, logits, labels, context_emb, answer_emb):
        bs = labels.shape[0]
        
        #losses
        primary_loss = F.cross_entropy(logits, labels)
        print(primary_loss)
        
        aux_loss = calculate_similarities(answer_emb, context_emb, labels, self.C, bs, self.device)
        
        print(aux_loss)
        #get the weights of the losses
        return primary_loss * (self.weight) + aux_loss * (1 - self.weight)
    
    
