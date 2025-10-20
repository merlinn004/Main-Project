from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# 4) Build pairs for a split
def build_pairs_for_split(common_keys: List[str],
                          audio_map: Dict[str, Path],
                          video_map: Dict[str, Path],
                          labels: Dict[str, int]):
    # Group by emotion
    by_emotion: Dict[int, List[str]] = defaultdict(list)
    for k in common_keys:
        by_emotion[labels[k]].append(k)

    pairs: List[Tuple[Path, Path, Path, Path, int]] = []

    # Positives: full cross-product within same emotion
    for emo, group in by_emotion.items():
        for s1 in group:
            for s2 in group:
                a1, v1 = audio_map[s1], video_map[s1]
                a2, v2 = audio_map[s2], video_map[s2]
                pairs.append((a1, v1, a2, v2, 1))
    # Negatives: across emotion groups
    emos = list(by_emotion.keys())
    for i in range(len(emos)):
        for j in range(len(emos)):
            if i == j:
                continue
            G1 = by_emotion[emos[i]]
            G2 = by_emotion[emos[j]]
            for s1 in G1:
                for s2 in G2:
                    a1, v1 = audio_map[s1], video_map[s1]
                    a2, v2 = audio_map[s2], video_map[s2]
                    pairs.append((a1, v1, a2, v2, 0))
    return pairs