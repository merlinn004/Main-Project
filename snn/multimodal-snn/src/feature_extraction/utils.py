import torch
import cv2
from PIL import Image
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import clip
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained models
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.eval()


def extract_audio_embedding_segment(waveform_segment_1d, sr):
    inputs = wav2vec_processor(waveform_segment_1d, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**{k: v.to(device) for k, v in inputs.items()})
    hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    embedding = hidden_states / hidden_states.norm()
    return embedding.float()


# Video feature extraction for segment (list of frames)
def extract_video_embedding_segment(frames):
    if len(frames) == 0:
        return None
    processed = torch.cat([clip_preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device) for f in frames])
    with torch.no_grad():
        feats = clip_model.encode_image(processed)
    feats = feats.float()
    feats /= feats.norm(dim=-1, keepdim=True)
    avg_feat = feats.mean(dim=0)
    avg_feat /= avg_feat.norm()
    return avg_feat.float()



def sliding_window_audio(y, sr, win_sec=1.0, hop_sec=0.5):
    win_samples = int(win_sec * sr)
    hop_samples = int(hop_sec * sr)
    segments = []
    n = len(y)
    if n < win_samples:
        pad = np.zeros(win_samples - n, dtype=y.dtype)
        seg = np.concatenate([y, pad], axis=0)
        segments.append(seg)
    else:
        for start in range(0, n - win_samples + 1, hop_samples):
            segments.append(y[start:start+win_samples])
        if len(segments) == 0:
            segments.append(y[-win_samples:])
    return segments

def sliding_window_video_frames(video_path: str, win_sec: float = 1.0, hop_sec: float = 0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    win_frames = int(round(win_sec * fps))
    hop_frames = int(round(hop_sec * fps))
    if win_frames <= 0:
        win_frames = 1
    if hop_frames <= 0:
        hop_frames = 1

    segments = []
    if len(frames) >= win_frames:
        for start in range(0, len(frames) - win_frames + 1, hop_frames):
            segments.append(frames[start:start + win_frames])
        if len(segments) == 0:
            segments.append(frames[-win_frames:])
    else:

        if len(frames) > 0:
            segments.append(frames)

    return segments