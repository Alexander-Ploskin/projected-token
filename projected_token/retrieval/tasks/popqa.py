from __future__ import annotations

from typing import Any


def build_popqa_cases(records: list[dict[str, Any]], valid_indices: list[int], question_col: str = "question") -> list[dict[str, Any]]:
    orig_to_vector = {orig: i for i, orig in enumerate(valid_indices)}
    cases = []
    for idx, record in enumerate(records):
        cases.append({
            "query": str(record.get(question_col, "")),
            "relevant_docs": {orig_to_vector[idx]} if idx in orig_to_vector else set(),
            "metadata": {"row_index": idx, "subj": record.get("subj"), "obj": record.get("obj")},
        })
    return cases
