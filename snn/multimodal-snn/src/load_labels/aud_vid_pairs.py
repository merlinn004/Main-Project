# file: src/load_labels/aud_vid_pairs_with_meta.py

from pathlib import Path
import csv
import logging
import sys
import math
import cv2
import librosa
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from tqdm import tqdm  # progress bar

# Local imports (package-relative)
from .audio_labels import load_meld_labels
from .norm import stem_to_du

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Probing helpers (fast, no full decode)
# -----------------------------------------------------------------------------
def probe_video_meta(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cv2.CAP_PROP_FRAME_COUNT and cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {"fps": float(fps), "frame_count": frame_count}

def estimate_segments(duration_sec: float, win_sec: float, hop_sec: float) -> int:
    if duration_sec <= 0 or win_sec <= 0 or hop_sec <= 0:
        return 0
    if duration_sec < win_sec:
        return 1
    return 1 + max(0, int(math.floor((duration_sec - win_sec) / hop_sec)))

# -----------------------------------------------------------------------------
# Main: build CSVs with meta per split
# -----------------------------------------------------------------------------
def build_audio_video_pairs_csvs(
    audio_base: str,
    video_base: str,
    emotion_pickle: str,
    out_base_dir: str,
    win_sec: float = 1.0,
    hop_sec: float = 0.5,
    include_meta: bool = True,
) -> None:
    """
    Writes {split}_pairs.csv with standard 5 columns, and if include_meta=True,
    writes {split}_pairs_with_meta.csv adding fps/win/hop/segment counts for each side.
    """
    audio_base_p = Path(audio_base)
    video_base_p = Path(video_base)
    out_base_p = Path(out_base_dir)
    out_base_p.mkdir(parents=True, exist_ok=True)

    # Load labels
    utt_id_to_label, _ = load_meld_labels(emotion_pickle)

    splits = ["train_splits", "dev_splits", "test_splits"]
    total_pairs = 0

    for split in splits:
        a_dir = audio_base_p / split
        v_dir = video_base_p / split
        logger.info(f"[{split}] Scanning split directories: A={a_dir}, V={v_dir}")

        # Build stem maps
        audio_files = sorted(a_dir.rglob("*.wav"))
        video_files = sorted(v_dir.rglob("*.mp4"))
        audio_map = {stem_to_du(p.stem): p for p in audio_files if stem_to_du(p.stem)}
        video_map = {stem_to_du(p.stem): p for p in video_files if stem_to_du(p.stem)}
        common = sorted(set(audio_map) & set(video_map) & set(utt_id_to_label))

        logger.info(f"[{split}] audio={len(audio_map)}, video={len(video_map)}, common={len(common)}")
        if not common:
            logger.warning(f"[{split}] No aligned items with labels; skipping.")
            continue

        # Probe meta per item
        item_meta: Dict[str, Dict[str, Any]] = {}
        bad_audio, bad_video = 0, 0

        for k in tqdm(common, desc=f"[{split}] Probing A/V meta", unit="item"):
            ap = str(audio_map[k])
            vp = str(video_map[k])

            try:
                adur = librosa.get_duration(path=ap)
            except Exception as e:
                bad_audio += 1
                logger.warning(f"[{split}] Failed to probe audio duration for {ap}: {e}")
                continue

            vinfo = probe_video_meta(vp)
            if vinfo is None:
                bad_video += 1
                logger.warning(f"[{split}] Failed to open video {vp}")
                continue

            fps = vinfo["fps"]
            frame_count = vinfo["frame_count"]
            vdur = (frame_count / fps) if frame_count > 0 else adur

            seg_audio = estimate_segments(adur, win_sec, hop_sec)
            seg_video = estimate_segments(vdur, win_sec, hop_sec)
            seg_aligned = max(0, min(seg_audio, seg_video))
            item_meta[k] = {
                "audio_path": ap,
                "video_path": vp,
                "label": int(utt_id_to_label[k]),
                "fps": float(fps),
                "win_sec": float(win_sec),
                "hop_sec": float(hop_sec),
                "seg_audio": int(seg_audio),
                "seg_video": int(seg_video),
                "seg_aligned": int(seg_aligned),
            }

        logger.info(f"[{split}] probed items={len(item_meta)}, bad_audio={bad_audio}, bad_video={bad_video}")
        valid = [k for k, m in item_meta.items() if m["seg_aligned"] > 0]
        logger.info(f"[{split}] valid items with ≥1 aligned segment: {len(valid)}")
        if not valid:
            logger.warning(f"[{split}] No valid items after probing; skipping.")
            continue

        # Group by emotion
        by_emotion: Dict[int, List[str]] = defaultdict(list)
        for k in valid:
            by_emotion[item_meta[k]["label"]].append(k)

        # Prepare writers
        out_csv = out_base_p / f"{split}_pairs.csv"
        f_std = out_csv.open("w", newline="", encoding="utf-8")
        w_std = csv.writer(f_std)
        w_std.writerow(["audio_path1", "video_path1", "audio_path2", "video_path2", "label"])

        if include_meta:
            out_meta = out_base_p / f"{split}_pairs_with_meta.csv"
            f_met = out_meta.open("w", newline="", encoding="utf-8")
            w_met = csv.writer(f_met)
            header = [
                "audio_path1","video_path1","audio_path2","video_path2","label",
                "fps1","win_sec1","hop_sec1","seg_aligned1",
                "fps2","win_sec2","hop_sec2","seg_aligned2",
            ]
            w_met.writerow(header)
        else:
            f_met = None
            w_met = None

        # Progress: estimate total pairs without self-duplicates
        # Positives total = sum over emotions of n_e * (n_e - 1) / 2
        pos_total = 0
        for emo, group in by_emotion.items():
            n = len(group)
            pos_total += n * (n - 1) // 2

        # Negatives: sum over unordered emotion pairs (e_i < e_j) of n_i * n_j,
        # but we will write both directions (i->j and j->i) to preserve original balance.
        # We'll count logical pairs as n_i * n_j and write 2 rows per logical pair.
        emo_keys = list(by_emotion.keys())
        neg_logical = 0
        for i in range(len(emo_keys)):
            for j in range(i + 1, len(emo_keys)):
                neg_logical += len(by_emotion[emo_keys[i]]) * len(by_emotion[emo_keys[j]])
        neg_total_rows = neg_logical * 2

        total_rows = pos_total + neg_total_rows
        pbar = tqdm(total=total_rows, desc=f"[{split}] Writing pairs", unit="pair")

        # Write positives: j > i (no self-pairs, no symmetric duplicates)
        for emo, group in by_emotion.items():
            g = group
            for i in range(len(g)):
                s1 = g[i]
                for j in range(i + 1, len(g)):
                    s2 = g[j]
                    a1, v1 = item_meta[s1]["audio_path"], item_meta[s1]["video_path"]
                    a2, v2 = item_meta[s2]["audio_path"], item_meta[s2]["video_path"]
                    # Standard
                    w_std.writerow([a1, v1, a2, v2, 1])
                    if w_met:
                        m1 = item_meta[s1]; m2 = item_meta[s2]
                        w_met.writerow([
                            a1, v1, a2, v2, 1,
                            m1["fps"], m1["win_sec"], m1["hop_sec"], m1["seg_aligned"],
                            m2["fps"], m2["win_sec"], m2["hop_sec"], m2["seg_aligned"],
                        ])
                    pbar.update(1)

        # Write negatives: for each unordered pair of emotion groups (i < j),
        # write both directions to roughly match previous balance: (s in Gi, t in Gj) and (t, s).
        for i in range(len(emo_keys)):
            for j in range(i + 1, len(emo_keys)):
                G1 = by_emotion[emo_keys[i]]
                G2 = by_emotion[emo_keys[j]]
                for s1 in G1:
                    m1 = item_meta[s1]
                    for s2 in G2:
                        m2 = item_meta[s2]
                        # Direction 1: s1 vs s2
                        w_std.writerow([m1["audio_path"], m1["video_path"], m2["audio_path"], m2["video_path"], 0])
                        if w_met:
                            w_met.writerow([
                                m1["audio_path"], m1["video_path"], m2["audio_path"], m2["video_path"], 0,
                                m1["fps"], m1["win_sec"], m1["hop_sec"], m1["seg_aligned"],
                                m2["fps"], m2["win_sec"], m2["hop_sec"], m2["seg_aligned"],
                            ])
                        pbar.update(1)
                        # Direction 2: s2 vs s1
                        w_std.writerow([m2["audio_path"], m2["video_path"], m1["audio_path"], m1["video_path"], 0])
                        if w_met:
                            w_met.writerow([
                                m2["audio_path"], m2["video_path"], m1["audio_path"], m1["video_path"], 0,
                                m2["fps"], m2["win_sec"], m2["hop_sec"], m2["seg_aligned"],
                                m1["fps"], m1["win_sec"], m1["hop_sec"], m1["seg_aligned"],
                            ])
                        pbar.update(1)

        pbar.close()
        f_std.close()
        if f_met:
            f_met.close()

        logger.info(f"[{split}] wrote pairs to {out_csv}" + (f" and {out_meta}" if include_meta else ""))
        total_pairs += total_rows

    logger.info(f"Done. Total pairs across splits: {total_pairs}")
