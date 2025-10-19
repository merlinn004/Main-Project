import torch
from typing import List, Dict, Any

def collate_av_pair_sequences(batch: List[Dict[str, Any]], pad_value: float = 0.0) -> Dict[str, Any]:
    # Find max segments in batch for each branch
    max_S1 = max(b["a1"].shape[0] for b in batch)
    max_S2 = max(b["a2"].shape[0] for b in batch)

    a1_list, v1_list, a2_list, v2_list, y_list, metas = [], [], [], [], [], []
    for b in batch:
        a1, v1 = b["a1"], b["v1"]  # [S, T, Fa], [S, T, Fv]
        a2, v2 = b["a2"], b["v2"]
        # Pad on S dimension
        def pad_S(x, maxS):
            S, T, F = x.shape
            if S == maxS:
                return x
            pad_tensor = torch.zeros((maxS - S, T, x.shape[2]), dtype=x.dtype)
            return torch.cat([x, pad_tensor.fill_(pad_value)], dim=0)
        a1_list.append(pad_S(a1, max_S1))
        v1_list.append(pad_S(v1, max_S1))
        a2_list.append(pad_S(a2, max_S2))
        v2_list.append(pad_S(v2, max_S2))
        y_list.append(int(b["label"]))
        metas.append(b["meta"])

    a1 = torch.stack(a1_list, dim=0)  # [B, S1, T, Fa]
    v1 = torch.stack(v1_list, dim=0)  # [B, S1, T, Fv]
    a2 = torch.stack(a2_list, dim=0)  # [B, S2, T, Fa]
    v2 = torch.stack(v2_list, dim=0)  # [B, S2, T, Fv]
    y = torch.tensor(y_list, dtype=torch.float32)  # contrastive expects float labels often
    return {"a1": a1, "v1": v1, "a2": a2, "v2": v2, "y": y, "meta": metas}
