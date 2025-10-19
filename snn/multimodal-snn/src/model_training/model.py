import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

import importlib
import src.feature_extraction.spike_gen as sg
importlib.reload(sg)
from src.feature_extraction.spike_gen import extract_aligned_spike_trains

# Ensure these functions are imported from your pipeline module
# from your_module import extract_aligned_spike_trains

class SpikingSiameseNetwork(nn.Module):
    def __init__(self, audio_dim, video_dim, encoder, embed_dim=128, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        self.encoder = encoder

        # Use the correct arg name for your snntorch version (spike_grad is most common)
        self.surrogate_func = surrogate.fast_sigmoid(slope=25)

        # Audio branch
        self.audio_fc1 = nn.Linear(audio_dim, 256)
        self.audio_lif1 = snn.Leaky(beta=0.9, spike_grad=self.surrogate_func)
        self.audio_fc2 = nn.Linear(256, embed_dim)
        self.audio_lif2 = snn.Leaky(beta=0.9, spike_grad=self.surrogate_func)

        # Video branch
        self.video_fc1 = nn.Linear(video_dim, 256)
        self.video_lif1 = snn.Leaky(beta=0.9, spike_grad=self.surrogate_func)
        self.video_fc2 = nn.Linear(256, embed_dim)
        self.video_lif2 = snn.Leaky(beta=0.9, spike_grad=self.surrogate_func)

    # Single-sample segment: audio_spikes/video_spikes: [T, feat]
    def forward_once_segment(self, audio_spikes, video_spikes):
        audio_spikes = audio_spikes.float()
        video_spikes = video_spikes.float()

        mem_audio1 = self.audio_lif1.init_leaky()
        mem_audio2 = self.audio_lif2.init_leaky()
        mem_video1 = self.video_lif1.init_leaky()
        mem_video2 = self.video_lif2.init_leaky()

        for t in range(self.num_steps):
            cur_audio = self.audio_fc1(audio_spikes[t])
            spk_audio1, mem_audio1 = self.audio_lif1(cur_audio, mem_audio1)
            cur_audio2 = self.audio_fc2(spk_audio1)
            spk_audio2, mem_audio2 = self.audio_lif2(cur_audio2, mem_audio2)

            cur_video = self.video_fc1(video_spikes[t])
            spk_video1, mem_video1 = self.video_lif1(cur_video, mem_video1)
            cur_video2 = self.video_fc2(spk_video1)
            spk_video2, mem_video2 = self.video_lif2(cur_video2, mem_video2)

        combined = torch.cat([spk_audio2, spk_video2], dim=0)
        combined = F.normalize(combined.unsqueeze(0), p=2, dim=1).squeeze(0)
        return combined

    # Single-sample sequence: audio_spikes_seq/video_spikes_seq: [S, T, feat]
    def forward_sequence(self, audio_spikes_seq, video_spikes_seq):
        embeddings = []
        num_segments = audio_spikes_seq.shape[0]
        for seg in range(num_segments):
            emb = self.forward_once_segment(audio_spikes_seq[seg], video_spikes_seq[seg])
            embeddings.append(emb.unsqueeze(0))
        embeddings = torch.cat(embeddings, dim=0)  # [S, embed_dim]
        final_emb = embeddings.mean(dim=0)         # [embed_dim]
        return final_emb

    # Batched segment: audio_spikes/video_spikes: [B, T, feat]
    def forward_once_segment_batched(self, audio_spikes, video_spikes):
        audio_spikes = audio_spikes.float()
        video_spikes = video_spikes.float()

        B, T, _ = audio_spikes.shape
        mem_audio1 = self.audio_lif1.init_leaky(batch_size=B)
        mem_audio2 = self.audio_lif2.init_leaky(batch_size=B)
        mem_video1 = self.video_lif1.init_leaky(batch_size=B)
        mem_video2 = self.video_lif2.init_leaky(batch_size=B)

        for t in range(T):
            cur_audio = self.audio_fc1(audio_spikes[:, t, :])
            spk_audio1, mem_audio1 = self.audio_lif1(cur_audio, mem_audio1)
            cur_audio2 = self.audio_fc2(spk_audio1)
            spk_audio2, mem_audio2 = self.audio_lif2(cur_audio2, mem_audio2)

            cur_video = self.video_fc1(video_spikes[:, t, :])
            spk_video1, mem_video1 = self.video_lif1(cur_video, mem_video1)
            cur_video2 = self.video_fc2(spk_video1)
            spk_video2, mem_video2 = self.video_lif2(cur_video2, mem_video2)

        combined = torch.cat([spk_audio2, spk_video2], dim=1)  # [B, 2*embed_dim]
        combined = F.normalize(combined, p=2, dim=1)           # [B, 2*embed_dim]
        return combined

    # Batched sequence: a_seq/v_seq: [B, S, T, feat]
    def forward_sequence_batched(self, a_seq, v_seq):
        a_seq = a_seq.float()
        v_seq = v_seq.float()

        B, S, T, F = a_seq.shape
        embs = []
        for s in range(S):
            emb = self.forward_once_segment_batched(a_seq[:, s], v_seq[:, s])  # [B, 2*embed_dim]
            embs.append(emb.unsqueeze(1))
        embs = torch.cat(embs, dim=1)  # [B, S, 2*embed_dim]
        final = embs.mean(dim=1)        # [B, 2*embed_dim]
        return final

    # Path-based forward for quick testing; still valid
    def forward(self, audio_path1, video_path1, audio_path2, video_path2):
        audio_spikes1, video_spikes1 = extract_aligned_spike_trains(audio_path1, video_path1, self.encoder)
        audio_spikes2, video_spikes2 = extract_aligned_spike_trains(audio_path2, video_path2, self.encoder)
        emb1 = self.forward_sequence(audio_spikes1, video_spikes1)
        emb2 = self.forward_sequence(audio_spikes2, video_spikes2)
        return emb1, emb2
