from typing import List, Tuple

import pandas as pd


def find_aligned_by_index(src_index: List[str], src_texts: List[str],
                          tgt_index: List[str], tgt_texts: List[str]) -> Tuple[List[str], List[str]]:
    df = pd.DataFrame.from_dict({"src_index": src_index, "src_text": src_texts,
                                 "tgt_index": tgt_index, "tgt_text": tgt_texts}, orient='index').T
    aligned_rows = df[df["src_index"] == df["tgt_index"]]
    return aligned_rows["src_text"].tolist(), aligned_rows["tgt_text"].tolist()
