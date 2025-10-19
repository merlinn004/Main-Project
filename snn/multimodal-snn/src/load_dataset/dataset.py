import sys
import logging
import time
from typing import Dict, Any, List, Tuple, Optional
import csv
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from src.feature_extraction.utils import (
    extract_audio_embedding_segment,
    extract_video_embedding_segment,
    sliding_window_audio,
    sliding_window_video_frames,  # used only if use_threaded_video=False
)
from src.feature_extraction.video_reader_parallel import ThreadedVideoReader, iter_frame_windows
from src.model_training.spike_encoder import DynamicSpikeEncoder

# ---------------------------------------------------------------------
# Logging configured to match your notebook style
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Configure only if not already configured by the notebook/app
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # default INFO; set DEBUG from notebook if needed

# ---------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------
encoder = DynamicSpikeEncoder(T=25, use_dynamic_encoding=True)

class AVPairSegmentsDataset(Dataset):
    """
    Supports both CSV schemas:
      - 5 columns: audio_path1, video_path1, audio_path2, video_path2, label
      - meta CSV: + fps1, win_sec1, hop_sec1, seg_aligned1, fps2, win_sec2, hop_sec2, seg_aligned2
    Returns:
      {'a1':[S1,T,Fa], 'v1':[S1,T,Fv], 'a2':[S2,T,Fa], 'v2':[S2,T,Fv], 'label': int, 'meta': {...}}
    """
    def __init__(
        self,
        csv_file: str,
        sr: int = 16000,
        win_sec: float = 1.0,
        hop_sec: float = 0.5,
        max_segments: Optional[int] = None,
        use_threaded_video: bool = True,
        show_sample_pbar: bool = False,  # enable only for debugging with num_workers=0
        verbose: bool = False,           # set True to emit DEBUG sample logs
    ):
        self.rows: List[Dict[str, Any]] = []
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)

        self.sr = sr
        self.default_win = win_sec
        self.default_hop = hop_sec
        self.max_segments = max_segments
        self.use_threaded_video = use_threaded_video
        self.show_sample_pbar = show_sample_pbar
        self.verbose = verbose

        # Elevate logger level if verbose requested
        if verbose and logger.level > logging.DEBUG:
            logger.setLevel(logging.DEBUG)

        logger.info(
            f"Dataset init: rows={len(self.rows)}, sr={sr}, default win/hop={win_sec}/{hop_sec}, "
            f"max_segments={max_segments}, threaded_video={use_threaded_video}, verbose={verbose}"
        )

    def _get_params(self, r: Dict[str, Any], side: int):
        win = float(r.get(f"win_sec{side}", self.default_win))
        hop = float(r.get(f"hop_sec{side}", self.default_hop))
        seg_cap = r.get(f"seg_aligned{side}")
        seg_aligned = int(seg_cap) if seg_cap is not None and str(seg_cap).isdigit() else None
        return win, hop, seg_aligned

    def _load_audio_segments(self, audio_path: str, win: float, hop: float) -> List[np.ndarray]:
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        return sliding_window_audio(y, self.sr, win, hop)

    def _iter_video_segments(self, video_path: str, win: float, hop: float):
        if not self.use_threaded_video:
            # Legacy, full-decode fallback
            frames_by_window = sliding_window_video_frames(video_path, win, hop)
            for frames in frames_by_window:
                yield frames
            return
        rdr = ThreadedVideoReader(video_path, queue_size=96, drop_oldest=True).start()
        try:
            for frames in iter_frame_windows(rdr, win_sec=win, hop_sec=hop):
                yield frames
        finally:
            rdr.stop()

    def _segments_to_spikes_av(
        self,
        audio_segments: List[np.ndarray],
        video_windows_iter,
        seg_aligned: Optional[int] = None,
        sample_desc: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        limit = seg_aligned if seg_aligned is not None else len(audio_segments)
        if self.max_segments is not None:
            limit = min(limit, self.max_segments)

        a_seq: List[torch.Tensor] = []
        v_seq: List[torch.Tensor] = []

        iterator = enumerate(video_windows_iter)
        if self.show_sample_pbar:
            iterator = tqdm(
                iterator,
                total=limit,
                desc=(sample_desc or "segments"),
                unit="seg",
                leave=False,
            )

        t0 = time.perf_counter()
        produced = 0
        for i, vid_seg in iterator:
            if i >= len(audio_segments):
                break
            if produced >= limit:
                break

            aud_seg = torch.tensor(audio_segments[i], dtype=torch.float32)
            if len(vid_seg) == 0:
                continue

            a_emb = extract_audio_embedding_segment(aud_seg, self.sr)
            v_emb = extract_video_embedding_segment(vid_seg)
            if a_emb is None or v_emb is None:
                continue

            a_spk = encoder.encode(a_emb)
            v_spk = encoder.encode(v_emb)
            a_seq.append(a_spk.unsqueeze(0))
            v_seq.append(v_spk.unsqueeze(0))
            produced += 1

        dt = time.perf_counter() - t0
        if self.verbose:
            logger.debug(f"{sample_desc or 'sample'}: produced {produced}/{limit} segments in {dt:.2f}s")

        if len(a_seq) == 0 or len(v_seq) == 0:
            raise RuntimeError("No valid aligned segments after encoding.")

        a_seq = torch.cat(a_seq, dim=0)  # [S, T, Fa]
        v_seq = torch.cat(v_seq, dim=0)  # [S, T, Fv]
        return a_seq, v_seq

    def _build_pair(self, audio_path: str, video_path: str, win: float, hop: float, seg_aligned: Optional[int], tag: str):
        a_segs = self._load_audio_segments(audio_path, win, hop)
        v_iter = self._iter_video_segments(video_path, win, hop)
        if self.verbose:
            logger.debug(f"{tag}: win={win}, hop={hop}, audio_seg_count={len(a_segs)}, seg_cap={seg_aligned}")
        return self._segments_to_spikes_av(a_segs, v_iter, seg_aligned=seg_aligned, sample_desc=tag)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        a1p, v1p = r["audio_path1"], r["video_path1"]
        a2p, v2p = r["audio_path2"], r["video_path2"]
        y = int(r["label"])

        win1, hop1, seg1 = self._get_params(r, 1)
        win2, hop2, seg2 = self._get_params(r, 2)

        t0 = time.perf_counter()
        a1, v1 = self._build_pair(a1p, v1p, win1, hop1, seg1, tag=f"idx{idx}-pair1")
        a2, v2 = self._build_pair(a2p, v2p, win2, hop2, seg2, tag=f"idx{idx}-pair2")
        dt = time.perf_counter() - t0

        if self.verbose:
            logger.debug(
                f"idx{idx}: label={y} | S1={a1.shape[0]}/{v1.shape[0]} "
                f"S2={a2.shape[0]}/{v2.shape[0]} | time={dt:.2f}s"
            )

        return {
            "a1": a1, "v1": v1,
            "a2": a2, "v2": v2,
            "label": y,
            "meta": {
                "a1": a1p, "v1": v1p, "a2": a2p, "v2": v2p,
                "win1": win1, "hop1": hop1, "seg1": seg1,
                "win2": win2, "hop2": hop2, "seg2": seg2,
            }
        }
