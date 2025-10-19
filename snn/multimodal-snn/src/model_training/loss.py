import torch.nn as nn
import torch.nn.functional as F
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, emb1, emb2, label):
        d = F.pairwise_distance(emb1.unsqueeze(0), emb2.unsqueeze(0))
        loss = label * d.pow(2) + (1 - label) * F.relu(self.margin - d).pow(2)
        return loss.mean()
