import librosa
import torch
import torch.nn.functional as F
from spikingjelly.clock_driven import encoding
import numpy as np
from scipy.signal import butter, filtfilt
from augment import AudioAugmentation


def preprocess_audio(file_path, n_mels=64, fmax=8000, T=100, encode_spikes=True, max_frames=400, augment=True):
    
    # Load and process audio
    y, sr = librosa.load(file_path, sr=22050)
    
    # pre-emphasis filter to balance frequency spectrum
    y_filter = librosa.effects.preemphasis(y, coef=0.97)
    
    # Silence Removal
    y_trimmed, _ = librosa.effects.trim(y_filter, top_db=20)

    # Normalization
    y_norm = librosa.util.normalize(y_trimmed)
    
    # apply augmentations
    if augment:
        aug = AudioAugmentation()
        y_aug = aug.apply_augmentation(torch.from_numpy(y_norm).float()).numpy()
    
    else:
        y_aug = y_norm
    
    # Mel-spectrogram
    S_mel = librosa.feature.melspectrogram(y=y_norm if not augment else y_aug, 
                                       sr=sr, 
                                       n_mels=n_mels, 
                                       fmax=fmax,
                                       n_fft=2048,
                                       hop_length=512,
                                       win_length=2048,
                                       window='hann',
                                       power=2.0)
    
    
    # convert to db
    S_db = librosa.power_to_db(S_mel, ref=np.max, top_db=80.0)
    
    # Better normalization for spike encoding
    S_norm = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-8)
    
    # Spectral centroid for frequency distribution
    spectral_centroid = librosa.feature.spectral_centroid(y=y_aug, sr=sr)
    spectral_centroid = np.repeat(spectral_centroid, n_mels, axis=0)
    
    # Combine features
    features = np.vstack([S_norm, spectral_centroid[:, :S_norm.shape[1]]])
    features = torch.from_numpy(features).float()
    
    # Pad or truncate
    if features.shape[1] < max_frames:
        pad = max_frames - features.shape[1]
        features = F.pad(features, (0, pad))
    else:
        features = features[:, :max_frames]
    
    if not encode_spikes:
        return features.detach().cpu().unsqueeze(0).unsqueeze(1).repeat(T, 1, 1, 1)         # shape [T, 1, n_mels, frames]

    return features