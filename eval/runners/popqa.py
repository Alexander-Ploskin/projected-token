from __future__ import annotations

from typing import Any
from collections import Counter
import pandas as pd

from eval.runners.base import BaseEvalRunner


class PopQAEvalRunner(BaseEvalRunner):
    """PopQA-specific."""

    def load_dataset(self) -> list[dict[str, Any]]:
        path = self.cfg.eval.dataset_path
        limit = self.cfg.eval.limit
        df = pd.read_parquet(path)
        if limit:
            df = df.head(limit)
        data = []
        for _, row in df.iterrows():
            gold = row['possible_answers']
            if isinstance(gold, str) and gold.startswith('['):
                gold = eval(gold)
            elif isinstance(gold, str):
                gold = [gold]
            data.append({
                'id': str(row.get('id', _)),
                'question': row['question'],
                'gold_answers': gold
            })
        return data

    def compute_row_metrics(self, prediction: str, gold_answers: list[str]) -> dict[str, float]:
        norm_pred = self.normalize_answer(prediction)
        norm_golds = [self.normalize_answer(g) for g in gold_answers]

        em = max(int(norm_pred == ng) for ng in norm_golds)

        f1_max = 0
        pred_tokens = norm_pred.split()
        for ng in norm_golds:
            gt_tokens = ng.split()
            if not pred_tokens or not gt_tokens:
                f1 = int(pred_tokens == gt_tokens)
            else:
                common = Counter(pred_tokens) & Counter(gt_tokens)
                num_same = sum(common.values())
                if num_same == 0:
                    f1 = 0
                else:
                    prec = num_same / len(pred_tokens)
                    rec = num_same / len(gt_tokens)
                    f1 = 2 * prec * rec / (prec + rec)
            f1_max = max(f1_max, f1)

        in_acc = max(int(ng in norm_pred) for ng in norm_golds)

        return {"em": float(em), "f1": f1_max, "in_acc": float(in_acc)}
