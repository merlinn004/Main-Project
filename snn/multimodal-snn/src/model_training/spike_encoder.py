import torch
import snntorch.spikegen as spikegen

class DynamicSpikeEncoder:
    def __init__(self, T=25, use_dynamic_encoding=True):
        self.T = T
        self.use_dynamic_encoding = use_dynamic_encoding

    def encode(self, features):
        features = features.to(dtype=torch.float32)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        signal_energy = torch.mean(features ** 2)
        if self.use_dynamic_encoding:
            dynamic_scale = torch.clamp(2.0 + signal_energy * 0.5, min=1.0, max=4.0)
            dynamic_bias = torch.clamp(0.05 + signal_energy * 0.02, min=0.01, max=0.1)
        else:
            dynamic_scale = 2.0
            dynamic_bias = 0.05
        probs = torch.clamp(features * dynamic_scale - dynamic_bias, min=0.0, max=1.0)
        spikes = spikegen.rate(probs, num_steps=self.T).float()
        return spikes