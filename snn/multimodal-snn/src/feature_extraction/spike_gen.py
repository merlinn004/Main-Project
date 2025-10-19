import librosa
import torch

from .utils import (
    extract_audio_embedding_segment,
    extract_video_embedding_segment,
    sliding_window_audio
)

from .video_reader_parallel import ThreadedVideoReader, iter_frame_windows


# keep existing extract_audio_embedding_segment, extract_video_embedding_segment

def extract_aligned_spike_trains(
    audio_path,
    video_path,
    encoder,
    win_sec=1.0,
    hop_sec=0.5,
    sr=16000,
    max_segments: int = None,   # optional cap
):
    # Audio segments (full waveform → window slices)
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    audio_segments = sliding_window_audio(y, sr, win_sec, hop_sec)

    # Video segments via threaded prefetcher + iterator
    rdr = ThreadedVideoReader(video_path, queue_size=96, drop_oldest=True).start()
    video_segments = []
    try:
        for frames in iter_frame_windows(rdr, win_sec=win_sec, hop_sec=hop_sec):
            video_segments.append(frames)
            if max_segments is not None and len(video_segments) >= max_segments:
                break
    finally:
        rdr.stop()

    # Align counts
    S = min(len(audio_segments), len(video_segments))
    if S == 0:
        raise RuntimeError(f"No aligned segments for {audio_path} and {video_path}")

    if max_segments is not None:
        S = min(S, max_segments)

    audio_spikes_seq, video_spikes_seq = [], []
    for i in range(S):
        aud_seg = torch.tensor(audio_segments[i], dtype=torch.float32)
        vid_seg = video_segments[i]
        if len(vid_seg) == 0:
            continue

        # Embeddings per segment
        audio_emb = extract_audio_embedding_segment(aud_seg, sr)
        video_emb = extract_video_embedding_segment(vid_seg)
        if audio_emb is None or video_emb is None:
            continue

        # Spike trains [T, F]
        audio_spikes = encoder.encode(audio_emb).float()
        video_spikes = encoder.encode(video_emb).float()
        audio_spikes_seq.append(audio_spikes.unsqueeze(0))  # [1, T, Fa]
        video_spikes_seq.append(video_spikes.unsqueeze(0))  # [1, T, Fv]

    if len(audio_spikes_seq) == 0 or len(video_spikes_seq) == 0:
        raise RuntimeError(f"No valid spike segments produced for {audio_path} and {video_path}")

    audio_spikes_seq = torch.cat(audio_spikes_seq, dim=0)  # [S, T, Fa]
    video_spikes_seq = torch.cat(video_spikes_seq, dim=0)  # [S, T, Fv]
    return audio_spikes_seq, video_spikes_seq
