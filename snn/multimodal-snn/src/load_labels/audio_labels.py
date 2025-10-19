import pickle

def load_meld_labels(pickle_path: str):
    with open(pickle_path, "rb") as f:
        revs, *rest = pickle.load(f)  # revs: list[dict], last: label_index
        label_index = rest[-1]
    utt_id_to_label = {}
    for utt in revs:
        # utt has keys: 'dialog', 'utterance', 'y' (string label)
        utt_id = f"{utt['dialog']}_{utt['utterance']}"  # e.g., "0_1"
        label_str = utt['y']
        # Some MELD dumps already store y as int; handle both
        if isinstance(label_str, str):
            label_idx = label_index[label_str]
        else:
            label_idx = int(label_str)
        utt_id_to_label[utt_id] = label_idx
    return utt_id_to_label, label_index  # label_index: {label_name: idx}