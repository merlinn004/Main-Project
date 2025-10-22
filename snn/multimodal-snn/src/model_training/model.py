import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikeAttentionNet(nn.Module):
    def __init__(self, in_dim, embed_dim, num_heads, num_classes, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, mask=None):
        """
        x: [B, T, F]
        mask: optional attention mask [B, T] (True = ignore)
        """
        x = self.embed(x)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)  # mask = [B, T]
        pooled = attn_out.mean(dim=1)
        logits = self.fc(pooled)
        return logits
