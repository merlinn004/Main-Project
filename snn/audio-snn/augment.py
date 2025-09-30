import torchaudio
import torch
import torch.nn.functional as F
import numpy as np

class AudioAugmentation:
    def __init__(self):
        self.augmentations = [
            self.add_gaussian_noise,
            self.time_stretch,
            self.pitch_shift,
            self.time_shift,
            self.speed_change
        ]
        
    def add_gaussian_noise(self, audio, snr_db=20):
        signal_power = torch.mean(audio**2)
        noise_power = signal_power / (10**(snr_db / 10))
        noise = torch.randn_like(audio) * torch.sqrt(noise_power)
        return audio + noise

    def time_stretch(self, audio, stretch_factor=None):
        if stretch_factor is None:
            stretch_factor = np.random.uniform(0.8, 1.2)
        # Convert waveform to spectrogram
        n_fft = 2048
        hop_length = 512
        # Use torchaudio.transforms.Spectrogram with power=None for complex output
        spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        spec = spec_transform(audio)
        # Apply time stretch
        stretched_spec = torchaudio.functional.time_stretch(spec, hop_length, stretch_factor)
        # Convert back to waveform
        stretched_audio = torch.istft(stretched_spec, n_fft=n_fft, hop_length=hop_length, length=audio.shape[-1])
        return stretched_audio

    def pitch_shift(self, audio, sr=22050, n_steps=None):
        if n_steps is None:
            n_steps = np.random.randint(-2, 3)
        return torchaudio.functional.pitch_shift(audio, sr, n_steps)

    def time_shift(self, audio, max_shift=0.2):
        shift = int(np.random.uniform(-max_shift, max_shift) * audio.shape[-1])
        return torch.roll(audio, shifts=shift)

    def speed_change(self, audio, speed_factor=None):
        if speed_factor is None:
            speed_factor = np.random.uniform(0.9, 1.1)
        # Resample to change speed
        orig_len = audio.shape[-1]
        new_len = int(orig_len / speed_factor)
        audio = F.interpolate(audio.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear', align_corners=False)
        audio = audio.squeeze()
        # Pad or trim to original length
        if audio.shape[-1] < orig_len:
            audio = F.pad(audio, (0, orig_len - audio.shape[-1]))
        else:
            audio = audio[:orig_len]
        return audio

    def apply_augmentation(self, audio, p=0.5):
        if np.random.random() < p:
            aug_func = np.random.choice(self.augmentations)
            return aug_func(audio)
        return audio