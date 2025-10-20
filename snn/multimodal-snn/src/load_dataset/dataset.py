import sys
import io
import os
import csv
import logging
from typing import Dict, Any, List, Optional, Iterator

import numpy as np
from torch.utils.data import Dataset

from src.feature_extraction.utils import sliding_window_audio, sliding_window_video_frames
from src.feature_extraction.video_reader_parallel import ThreadedVideoReader, iter_frame_windows

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    f = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    h.setFormatter(f)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


class AVPairSegmentsDataset(Dataset):
    """
    Streams CSV by byte offsets. Returns raw audio windows and video frame lists per segment.
    No embeddings or spikes here.
    """
    def __init__(
        self,
        csv_file: str,
        sr: int = 16000,
        win_sec: float = 1.0,
        hop_sec: float = 0.5,
        max_segments: Optional[int] = None,
        use_threaded_video: bool = True,
        video_queue_size: int = 256,
        audio_loader: str = "librosa",  # or "torchaudio"
        verbose: bool = False,
    ):
        self.csv_file = csv_file
        self.sr = sr
        self.default_win, self.default_hop = win_sec, hop_sec
        self.max_segments = max_segments
        self.use_threaded_video = use_threaded_video
        self.video_queue_size = video_queue_size
        self.audio_loader = audio_loader
        self.verbose = verbose

        # Byte-offset index
        self._fh = open(self.csv_file, "r", newline="", encoding="utf-8")
        self._fh.seek(0)
        self._header_line = next(self._fh)
        self._offsets: List[int] = []
        pos = self._fh.tell()
        for _ in self._fh:
            self._offsets.append(pos)
            pos = self._fh.tell()

        if verbose:
            logger.setLevel(logging.DEBUG)
        logger.info(f"Dataset (raw) init: rows={len(self._offsets)}, sr={sr}, win/hop={win_sec}/{hop_sec}")

    def __len__(self) -> int:
        return len(self._offsets)

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass

    def _read_row(self, idx: int) -> Dict[str, Any]:
        self._fh.seek(self._offsets[idx])
        line = self._fh.readline()
        reader = csv.DictReader(io.StringIO(self._header_line + line))
        return next(reader)

    def _get_params(self, r: Dict[str, Any], side: int):
        win = float(r.get(f"win_sec{side}", self.default_win))
        hop = float(r.get(f"hop_sec{side}", self.default_hop))
        cap = r.get(f"seg_aligned{side}")
        seg_aligned = int(cap) if cap and cap.isdigit() else None
        return win, hop, seg_aligned

    def _load_audio_1d(self, path: str) -> np.ndarray:
        if self.audio_loader == "torchaudio":
            import torchaudio as ta, torchaudio.functional as AF
            wav, sr0 = ta.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr0 != self.sr:
                wav = AF.resample(wav, sr0, self.sr)
            return wav.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            import librosa
            y, _ = librosa.load(path, sr=self.sr, mono=True)
            return y.astype(np.float32)

    def _audio_segments(self, path: str, win: float, hop: float) -> List[np.ndarray]:
        y = self._load_audio_1d(path)
        segs = sliding_window_audio(y, self.sr, win, hop)
        return segs[: self.max_segments] if self.max_segments else segs

    def _iter_video(self, path: str, win: float, hop: float) -> Iterator[List[np.ndarray]]:
        if not self.use_threaded_video:
            for frames in sliding_window_video_frames(path, win, hop):
                yield frames
            return
        rdr = ThreadedVideoReader(path, queue_size=self.video_queue_size, drop_oldest=True).start()
        try:
            for frames in iter_frame_windows(rdr, win_sec=win, hop_sec=hop):
                yield frames
        finally:
            rdr.stop()

    def _collect_video(self, path: str, win: float, hop: float, limit: Optional[int]) -> List[List[np.ndarray]]:
        vids = []
        for i, frames in enumerate(self._iter_video(path, win, hop)):
            if not frames:
                continue
            vids.append(frames)
            if limit and len(vids) >= limit:
                break
        return vids

    def _build_pair(self, a_path, v_path, win, hop, aligned, tag):
        a_segs = self._audio_segments(a_path, win, hop)
        limit = aligned if aligned else len(a_segs)
        if self.max_segments:
            limit = min(limit, self.max_segments)
        v_segs = self._collect_video(v_path, win, hop, limit)
        S = min(len(a_segs), len(v_segs), limit)
        return a_segs[:S], v_segs[:S]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self._read_row(idx)
        a1p, v1p = r["audio_path1"], r["video_path1"]
        a2p, v2p = r["audio_path2"], r["video_path2"]
        y = int(r["label"])
        w1, h1, s1 = self._get_params(r, 1)
        w2, h2, s2 = self._get_params(r, 2)

        a1_raw, v1_raw = self._build_pair(a1p, v1p, w1, h1, s1, f"idx{idx}-p1")
        a2_raw, v2_raw = self._build_pair(a2p, v2p, w2, h2, s2, f"idx{idx}-p2")

        return {
            "a1_raw": a1_raw,
            "v1_raw": v1_raw,
            "a2_raw": a2_raw,
            "v2_raw": v2_raw,
            "label": y,
            "meta": {"a1": a1p, "v1": v1p, "a2": a2p, "v2": v2p},
        }
