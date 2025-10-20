import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from typing import List

from src.feature_extraction.utils import extract_audio_embedding_segment, extract_video_embedding_segment


class SpikingSiameseNetwork(nn.Module):
    def __init__(self, audio_dim, video_dim, encoder, embed_dim=128, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        self.encoder = encoder

        self.surrogate = surrogate.fast_sigmoid(slope=25)
        # Audio branch
        self.audio_fc1 = nn.Linear(audio_dim, 256)
        self.audio_lif1 = snn.Leaky(beta=0.9, spike_grad=self.surrogate)
        self.audio_fc2 = nn.Linear(256, embed_dim)
        self.audio_lif2 = snn.Leaky(beta=0.9, spike_grad=self.surrogate)
        # Video branch
        self.video_fc1 = nn.Linear(video_dim, 256)
        self.video_lif1 = snn.Leaky(beta=0.9, spike_grad=self.surrogate)
        self.video_fc2 = nn.Linear(256, embed_dim)
        self.video_lif2 = snn.Leaky(beta=0.9, spike_grad=self.surrogate)

    def extract_embeddings_and_spikes_batch(self, batch: List[dict], sr: int, device: torch.device):
        """
        Converts raw segments in batch → spike tensors [B, S, T, F].
        Returns a1, v1, a2, v2 (torch.Tensor) and labels (torch.Tensor).
        """
        B = len(batch)
        a1_list, v1_list, a2_list, v2_list, labels = [], [], [], [], []

        for sample in batch:
            # Pair 1
            a1_spk = self._to_spikes(sample["a1_raw"], sr)
            v1_spk = self._to_spikes(sample["v1_raw"], sr, is_audio=False)
            # Pair 2
            a2_spk = self._to_spikes(sample["a2_raw"], sr)
            v2_spk = self._to_spikes(sample["v2_raw"], sr, is_audio=False)

            a1_list.append(a1_spk)
            v1_list.append(v1_spk)
            a2_list.append(a2_spk)
            v2_list.append(v2_spk)
            labels.append(sample["label"])

        # Pad to max segments per pair
        max_S1 = max(t.shape[0] for t in a1_list)
        max_S2 = max(t.shape[0] for t in a2_list)

        a1 = self._pad_and_stack(a1_list, max_S1, device)
        v1 = self._pad_and_stack(v1_list, max_S1, device)
        a2 = self._pad_and_stack(a2_list, max_S2, device)
        v2 = self._pad_and_stack(v2_list, max_S2, device)
        labels = torch.tensor(labels, dtype=torch.float32, device=device)

        return a1, v1, a2, v2, labels

    def _to_spikes(self, raw_segs: List, sr: int, is_audio: bool = True) -> torch.Tensor:
        spikes = []
        for seg in raw_segs:
            if is_audio:
                emb = extract_audio_embedding_segment(torch.tensor(seg), sr)
            else:
                emb = extract_video_embedding_segment(seg)
            if emb is None:
                continue
            spk = self.encoder.encode(emb)  # [T, F]
            spikes.append(spk.unsqueeze(0))  # [1, T, F]
        if not spikes:
            F = self.audio_dim if is_audio else self.video_dim
            return torch.zeros((1, self.num_steps, F), dtype=torch.float32)
        return torch.cat(spikes, dim=0)  # [S, T, F]

    def _pad_and_stack(self, tensors: List[torch.Tensor], max_S: int, device: torch.device):
        padded = []
        for t in tensors:
            S, T, F = t.shape
            if S < max_S:
                pad = torch.zeros((max_S - S, T, F), dtype=t.dtype, device=device)
                t = torch.cat([t, pad], dim=0)
            padded.append(t.unsqueeze(0).to(device))
        return torch.cat(padded, dim=0)  # [B, max_S, T, F]

    def forward_once_segment_batched(self, a_spk, v_spk):
        """[B, T, F] → [B, 2*embed_dim]"""
        a_spk, v_spk = a_spk.float(), v_spk.float()
        B, T, _ = a_spk.shape
        mem_a1 = self.audio_lif1.init_leaky(batch_size=B)
        mem_a2 = self.audio_lif2.init_leaky(batch_size=B)
        mem_v1 = self.video_lif1.init_leaky(batch_size=B)
        mem_v2 = self.video_lif2.init_leaky(batch_size=B)

        for t in range(T):
            x = self.audio_fc1(a_spk[:, t]); s1, mem_a1 = self.audio_lif1(x, mem_a1)
            x2 = self.audio_fc2(s1); s2, mem_a2 = self.audio_lif2(x2, mem_a2)
            y = self.video_fc1(v_spk[:, t]); t1, mem_v1 = self.video_lif1(y, mem_v1)
            y2 = self.video_fc2(t1); t2, mem_v2 = self.video_lif2(y2, mem_v2)

        out = torch.cat([s2, t2], dim=1)
        return F.normalize(out, p=2, dim=1)

    def forward_sequence_batched(self, a_seq, v_seq):
        """[B, S, T, F] → [B, 2*embed_dim]"""
        B, S, _, _ = a_seq.shape
        embs = []
        for s in range(S):
            embs.append(self.forward_once_segment_batched(a_seq[:, s], v_seq[:, s]).unsqueeze(1))
        embs = torch.cat(embs, dim=1)
        return embs.mean(dim=1)
