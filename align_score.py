#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List
from tqdm import tqdm

from alignscore import AlignScore  # pip-installed package


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON on line {i}: {e}") from e


def write_jsonl(path: str, records):
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input .jsonl path")
    ap.add_argument("--output", required=True, help="Output .jsonl path")
    ap.add_argument("--ckpt", required=True, help="Path to AlignScore checkpoint, e.g. ./AlignScore-large.ckpt")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-mode", default="nli_sp", choices=["nli_sp", "nli", "bin_sp", "bin"])
    ap.add_argument("--batch", type=int, default=64, help="JSONL batching for throughput (independent of model batch_size)")
    args = ap.parse_args()

    scorer = AlignScore(
        model=args.model,
        batch_size=args.batch_size,
        device=args.device,
        ckpt_path=args.ckpt,
        evaluation_mode=args.eval_mode,
    )  # AlignScore usage per repo docs [web:101]

    out: List[Dict[str, Any]] = []

    ctx_buf, clm_buf, obj_buf = [], [], []

    def flush():
        nonlocal ctx_buf, clm_buf, obj_buf, out
        if not obj_buf:
            return
        scores = scorer.score(contexts=ctx_buf, claims=clm_buf)  # returns list-like [web:101]
        for obj, s in zip(obj_buf, scores):
            obj2 = dict(obj)
            obj2["align_score"] = float(s)
            out.append(obj2)
        ctx_buf, clm_buf, obj_buf = [], [], []

    for line_no, obj in tqdm(iter_jsonl(args.input)):
        context = obj.get("s_wiki_content", "")
        claim = obj.get("rephrased_text", "")

        if not isinstance(context, str) or not isinstance(claim, str):
            raise RuntimeError(f"Line {line_no}: expected string fields s_wiki_content/rephrased_text")

        ctx_buf.append(context)
        clm_buf.append(claim)
        obj_buf.append(obj)

        if len(obj_buf) >= args.batch:
            flush()

    flush()
    write_jsonl(args.output, out)


if __name__ == "__main__":
    main()
