# 2) Normalizers from file stems to MELD key "d_u"
def stem_to_du(stem: str) -> str:
    """
    Convert a filename stem to 'dialog_utterance' like '0_1'.
    Supports:
      - 'dia0_utt1' -> '0_1'
      - '0_1' -> '0_1'
      - 'dia000_utt001' -> '0_1'
    """
    if stem.startswith("dia") and "_utt" in stem:
        try:
            dialog_num = stem.split("dia", 1)[1].split("_utt", 1)[0]
            utt_num = stem.split("_utt", 1)[1]
            # remove leading zeros
            d = str(int(dialog_num))
            u = str(int(utt_num))
            return f"{d}_{u}"
        except Exception:
            return ""
    # Already "d_u" form
    if "_" in stem:
        parts = stem.split("_")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{int(parts[0])}_{int(parts[1])}"
    return ""