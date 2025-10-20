import logging
import torchaudio
import numpy as np
from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_audio_spectrogram(path):
    """
    Load an audio file and compute its Mel spectrogram.

    Args:
        path (str or pathlib.Path): Path to the audio file.

    Returns:
        torch.Tensor: Mel spectrogram tensor of shape [n_mels, time].
    """
    waveform, sample_rate = torchaudio.load(str(path))
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64)
    mel_spec = transform(waveform).squeeze(0)
    return mel_spec

def extract_heuristic_features(mel_spec):
    """
    Extract heuristic features from a Mel spectrogram.

    Features include total energy, duration (time frames),
    low frequency energy and high frequency energy.

    Args:
        mel_spec (torch.Tensor): Mel spectrogram tensor.

    Returns:
        list of float: Feature vector.
    """
    total_energy = mel_spec.sum().item()
    duration = mel_spec.shape[1]
    low_freq_energy = mel_spec[:16, :].sum().item()
    high_freq_energy = mel_spec[48:, :].sum().item()
    return [total_energy, duration, low_freq_energy, high_freq_energy]

def detect_outliers(train_data, contamination=0.05, random_state=42):
    """
    Perform outlier detection on audio spectrogram data.

    Args:
        train_data (tuple): (paths_list, labels_list)
        contamination (float): assumed proportion of outliers in the data.
        random_state (int): random seed for reproducibility.

    Returns:
        list of tuples: filtered (path, label) pairs after removing outliers.
    """
    paths, labels = train_data
    features = []

    logger.info(f"Extracting heuristic features from {len(paths)} samples")
    for path in paths:
        try:
            mel_spec = load_audio_spectrogram(path)
            feat = extract_heuristic_features(mel_spec)
            features.append(feat)
        except Exception as e:
            logger.warning(f"Failed to process {path}: {e}")

    features_np = np.array(features)
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(features_np)
    outlier_flags = clf.predict(features_np)  # -1 is outlier, 1 is inlier

    filtered = [
        (path, label) for (path, label), flag in zip(zip(paths, labels), outlier_flags) if flag == 1
    ]

    logger.info(f"Number of samples before filtering: {len(paths)}")
    logger.info(f"Number of samples after filtering: {len(filtered)}")

    return filtered
