from typing import List, Dict, Any

def collate_raw_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns the batch as a list of sample dicts.
    Embedding extraction and spike encoding are done in the training step.
    """
    return batch
