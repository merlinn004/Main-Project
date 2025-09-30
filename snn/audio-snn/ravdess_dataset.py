import os
import numpy as np
import torch
from torch.utils.data import Dataset
import preprocess
import snntorch.spikegen as spikegen


class RAVDESSDataset(Dataset):
    """
    Minimal RAVDESS dataset.

    Assumes preprocess.preprocess_audio(...) returns a tensor with shape:
        [T, 1, n_mels, time_frames]
    (i.e. already expanded across time and channel).
    """

    def __init__(
        self,
        root_dir: str,
        T: int = 100,
        n_mels: int = 64,
        fmax: int = 8000,
        transform = None,
        max_frames: int = 400,
        augment_prob: float = 0.5,
        use_dynamic_encoding = True
    ):
        self.root_dir = root_dir
        self.T = T
        self.n_mels = n_mels
        self.fmax = fmax
        self.transform = transform
        self.max_frames = max_frames
        self.augment_prob = augment_prob
        self.use_dynamic_encoding = use_dynamic_encoding

        # loading file paths and labels
        self.file_paths = []
        self.labels = []

        for actor_dir in sorted(os.listdir(root_dir)):
            actor_path = os.path.join(root_dir, actor_dir)
            if not os.path.isdir(actor_path):
                continue
            
            for fname in sorted(os.listdir(actor_path)):
                if not fname.lower().endswith(".wav"):
                    continue
                
                # expect filename format like "03-01-XX-XX-XX-XX-XX.wav"
                
                try:
                    emotion_id = int(fname.split("-")[2])
                except Exception:
                    continue
                
                self.file_paths.append(os.path.join(actor_path, fname))
                self.labels.append(emotion_id - 1)  # zero-based label

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        label = int(self.labels[idx])

        # Preprocessing
        is_training = self.transform is not None
        features = preprocess.preprocess_audio(
            file_path,
            n_mels=self.n_mels,
            fmax=self.fmax,
            T=self.T,
            encode_spikes=False,
            max_frames=self.max_frames,
            augment=is_training and np.random.random() < self.augment_prob
        )
        
        # Fix shape if needed
        if features.dim() == 4 and features.shape[0] == self.T:
            features = features[0]  # Take first time slice
        elif features.dim() == 5:
            features = features.squeeze(0)

            
        # normalize
        features = (features - features.min()) / (features.max() - features.min())
        
        # dynamic spike encoding
        if self.use_dynamic_encoding:
            # adaptive scaling based on energy of the input
            signal_energy = torch.mean(features**2)
            dynamic_scale = torch.clamp(2.0+signal_energy*0.5,1.0,4.0)
            dynamic_bias = torch.clamp(0.05+signal_energy*0.02,0.01,0.1)
            
        else:
            dynamic_scale = 2.0
            dynamic_bias = 0.05

        # Apply scaling and bias
        probs = torch.clamp(features * dynamic_scale - dynamic_bias, 0.0, 1.0)

        # Rate coding using snntorch.spikegen
        spikes = spikegen.rate(probs, num_steps=self.T)  # shape [T, 1, n_mels, time_frames]

        if self.transform:
            spikes = self.transform(spikes)
        return spikes, torch.tensor(label, dtype=torch.long)