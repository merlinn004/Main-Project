# file: src/load_labels/audio_align.py

from pathlib import Path
from typing import Dict, List, Tuple
import logging
import sys
from .norm import stem_to_du

# Configure logging similar to your notebook
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def align_audio_with_labels(
    audio_file_paths: List[Path],
    utt_id_to_label: Dict[str, int],
) -> Tuple[List[str], List[int]]:
    """
    For a list of audio file paths, return lists of aligned paths and labels.
    Only files whose normalized stem is present in utt_id_to_label are kept.
    """
    aligned_paths: List[str] = []
    aligned_labels: List[int] = []

    missing_label = 0
    bad_name = 0

    for p in audio_file_paths:
        stem = p.stem
        du = stem_to_du(stem)
        if not du:
            bad_name += 1
            logger.debug(f"Skipping file with unexpected name format: {stem}")
            continue

        label = utt_id_to_label.get(du)
        if label is None:
            missing_label += 1
            logger.debug(f"No label for {du} (file {stem})")
            continue

        aligned_paths.append(str(p))
        aligned_labels.append(int(label))

    if bad_name:
        logger.info(f"Skipped {bad_name} files due to unrecognized naming.")
    if missing_label:
        logger.info(f"Skipped {missing_label} files due to missing labels.")

    return aligned_paths, aligned_labels

def split_and_align_data(
    base_folder: str,
    utt_id_to_label: Dict[str, int],
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """
    Scan data under base_folder/{train_splits,dev_splits,test_splits}, align audio .wav files
    to utt_id_to_label using normalized stems, and return:
      train_data = (train_paths, train_labels)
      dev_data   = (dev_paths, dev_labels)
      test_data  = (test_paths, test_labels)
    """
    base_path = Path(base_folder)
    splits = ["train_splits", "dev_splits", "test_splits"]

    results = {}
    for split in splits:
        split_dir = base_path / split
        if not split_dir.exists():
            logger.warning(f"Split folder not found: {split_dir}")
            results[split] = ([], [])
            continue

        audio_files = sorted(split_dir.rglob("*.wav"))
        logger.info(f"[{split}] Found {len(audio_files)} audio files")

        paths, labels = align_audio_with_labels(audio_files, utt_id_to_label)
        logger.info(f"[{split}] Aligned {len(paths)} with labels")

        results[split] = (paths, labels)

    train_data = results.get("train_splits", ([], []))
    dev_data = results.get("dev_splits", ([], []))
    test_data = results.get("test_splits", ([], []))

    return train_data, dev_data, test_data
