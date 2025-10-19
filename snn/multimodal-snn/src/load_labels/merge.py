from pathlib import Path
import logging
import sys
from typing import Dict

from .norm import stem_to_du

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# 3) Scan split directories and align by stems
def collect_split_items(audio_split_dir: Path, video_split_dir: Path, utt_id_to_label: Dict[str, int]):
    audio_files = sorted(audio_split_dir.rglob("*.wav"))
    video_files = sorted(video_split_dir.rglob("*.mp4"))

    audio_map = {}
    for p in audio_files:
        du = stem_to_du(p.stem)
        if du:
            audio_map[du] = p
        else:
            logger.debug(f"Unrecognized audio stem format: {p.stem}")

    video_map = {}
    for p in video_files:
        du = stem_to_du(p.stem)
        if du:
            video_map[du] = p
        else:
            logger.debug(f"Unrecognized video stem format: {p.stem}")

    # Keep only keys present in both modalities and with labels
    common = set(audio_map) & set(video_map) & set(utt_id_to_label)

    dropped_audio_only = set(audio_map) - common
    dropped_video_only = set(video_map) - common
    dropped_no_label = (set(audio_map) & set(video_map)) - common

    return audio_map, video_map, utt_id_to_label, sorted(common), dropped_audio_only, dropped_video_only, dropped_no_label