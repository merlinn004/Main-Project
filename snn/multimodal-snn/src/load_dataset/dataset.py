import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import snntorch.spikegen as spikegen
from collections import Counter
from sklearn.model_selection import train_test_split


# ------------------------------
# Augmentation utilities
# ------------------------------
def temporal_jitter(spike_train, max_shift=2):
    """Shift spikes randomly along time axis."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        spike_train = np.pad(spike_train[:-shift], ((shift, 0), (0, 0)))
    elif shift < 0:
        spike_train = np.pad(spike_train[-shift:], ((0, -shift), (0, 0)))
    return spike_train


def add_noise(spike_train, noise_level=0.05):
    """Add small random noise."""
    noise = np.random.randn(*spike_train.shape) * noise_level
    return np.clip(spike_train + noise, 0, 1)


def random_mask(spike_train, mask_prob=0.1):
    """Randomly mask a percentage of time steps."""
    mask = np.ones(spike_train.shape[0], dtype=bool)
    mask_indices = np.random.rand(spike_train.shape[0]) < mask_prob
    mask[mask_indices] = False
    spike_train[~mask] = 0
    return spike_train, mask


# ------------------------------
# Dataset class
# ------------------------------
class MELDAudioSpikesAugmented(Dataset):
    """
    Simplified dataset with:
    - Spike train generation
    - Optional augmentation (jitter, noise, masking)
    """

    def __init__(
        self,
        features_path="data/features/audio_embeddings_feature_selection_emotion.pkl",
        labels_path="data/features/data_emotion.p",
        T=25,
        spike_cap=1.0,
        normalize=True,
        augment=False,
    ):
        super().__init__()
        self.T = T
        self.augment = augment

        # ---- Load features ----
        with open(features_path, "rb") as f:
            train_emb, val_emb, test_emb = pickle.load(f)
        merged_features = {**train_emb, **val_emb, **test_emb}

        # ---- Load labels ----
        with open(labels_path, "rb") as f:
            data_list = pickle.load(f)
            utter_list = data_list[0]
            label_idx = data_list[5]

        feats, labels = [], []
        for u in utter_list:
            key = f"{u['dialog']}_{u['utterance']}"
            if key in merged_features:
                x = merged_features[key].astype(np.float32)
                if normalize:
                    x = x / (np.linalg.norm(x) + 1e-8)
                x = np.clip(x, 0, spike_cap)
                feats.append(x)
                labels.append(label_idx[u["y"]])

        self.X = np.stack(feats)
        self.y = np.array(labels, dtype=np.int64)
        print(f"Loaded simplified spike dataset: N={len(self.X)}, F={self.X.shape[1]}")

        # Class distribution
        print("Class distribution:", Counter(self.y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # Convert to spike train
        S = spikegen.rate(x, num_steps=self.T).astype(np.float32)  # [T, F]

        mask = np.ones(self.T, dtype=bool)
        if self.augment:
            S = temporal_jitter(S)
            S = add_noise(S)
            S, mask = random_mask(S)

        return torch.tensor(S, dtype=torch.float32), torch.tensor(y, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


# ------------------------------
# Loader creation with oversampling + split
# ------------------------------
def create_balanced_loader(
    features_path,
    labels_path,
    batch_size=64,
    augment=True,
    T=25,
    val_split=0.2,
):
    full_dataset = MELDAudioSpikesAugmented(
        features_path=features_path,
        labels_path=labels_path,
        augment=augment,
        T=T,
    )

    # ---- Train/Val split ----
    train_idx, val_idx = train_test_split(
        np.arange(len(full_dataset)),
        test_size=val_split,
        stratify=full_dataset.y,
        random_state=42,
    )

    train_X, val_X = full_dataset.X[train_idx], full_dataset.X[val_idx]
    train_y, val_y = full_dataset.y[train_idx], full_dataset.y[val_idx]

    # ---- Create datasets ----
    train_set = MELDAudioSpikesAugmented(features_path, labels_path, T, augment=True)
    val_set = MELDAudioSpikesAugmented(features_path, labels_path, T, augment=False)
    train_set.X, train_set.y = train_X, train_y
    val_set.X, val_set.y = val_X, val_y

    # ---- Oversampling to fix imbalance ----
    class_counts = Counter(train_set.y)
    weights = [1.0 / class_counts[c] for c in train_set.y]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_set.y), replacement=True)

    print("Train class distribution:", Counter(train_set.y))
    print("Val class distribution:", Counter(val_set.y))

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader
