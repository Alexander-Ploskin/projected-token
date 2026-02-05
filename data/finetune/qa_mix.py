from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import datasets


# ----------------------------
# Templates
# ----------------------------

TEMPLATES_FOR_QA = [
    "Question: {question}?\nAnswer:",
    "{question}?",
    "Answer the following question:\n\n{question}",
    "Answer this question:\n\n{question}?",
    "Please answer this question: {question}",
    "Answer the question...{question}?",
    "What is the answer to this question? {question}\n\n",
    "Can you tell me the answer to {question}?",
    "Next question: {question}\n\n",
    "Q: {question} A:",
    "{question}\nWhat is the answer?",
    "Write the answer: {question}",
    "{question}???",
]

TEMPLATES_FOR_SUM = [
    "Write a short summary for the text\n\nSummary:",
    "Briefly summarize this article:\nSummary:",
    "What is a shorter version of this:\n\nSummary:",
    "Write a brief summary in a sentence or less.",
    "What is a very short summary of the above text?",
    "Summarize the aforementioned text in a single phrase.",
    "Can you generate a short summary of the above paragraph?",
    "Summarize the above articles\n\ntl;dr:",
]

TEMPLATE_FOR_FACT_CHECKING = [
    'Verify the following claims with "True" or "False":\n{question}',
]


# ----------------------------
# Shared helpers
# ----------------------------

def _print_header(name: str) -> None:
    print(f"\n=== {name} ===")


def _maybe_select(ds: datasets.Dataset, max_examples: int | None) -> datasets.Dataset:
    if max_examples is None:
        return ds
    max_examples = max(0, int(max_examples))
    if len(ds) <= max_examples:
        return ds
    return ds.select(range(max_examples))


def _safe_first(xs: Any, default: str = "") -> str:
    if xs is None:
        return default
    if isinstance(xs, (list, tuple)) and len(xs) > 0:
        return xs[0] if xs[0] is not None else default
    return default


def _format_qa_question(question: str, rng: random.Random) -> str:
    question = (question or "").strip()
    return rng.choice(TEMPLATES_FOR_QA).format_map({"question": question})


def _format_sum_prompt(rng: random.Random) -> str:
    return rng.choice(TEMPLATES_FOR_SUM)


def _format_fact_prompt(claims: str) -> str:
    return TEMPLATE_FOR_FACT_CHECKING[0].format_map({"question": (claims or "").strip()})


def _mk_messages(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _mk_record(
    _id: str,
    task_type: str,
    messages: list[dict[str, str]],
    background: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"id": _id, "task_type": task_type, "messages": messages}
    if background is not None:
        out["background"] = background
    if meta:
        out["meta"] = meta
    return out


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(rows: Iterable[dict[str, Any]], path: str) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ----------------------------
# Dataset loaders (all same style)
# ----------------------------

def load_commonsense_qa(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("commonsense_qa")
    ds = datasets.load_dataset("commonsense_qa", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        q = (sample["question"] or "").strip() + "\n\n"
        for choice, text in zip(sample["choices"]["label"], sample["choices"]["text"]):
            q += f"{choice}. {text}\n"
        user = _format_qa_question(q, rng)
        answer = (sample["answerKey"] or "").strip()

        out.append(_mk_record(
            _id=f"commonsense_qa_{i}",
            task_type="open_qa",
            messages=_mk_messages(user, answer),
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_web_questions(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("web_questions")
    ds = datasets.load_dataset("web_questions", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        user = _format_qa_question(sample["question"], rng)
        answer = _safe_first(sample.get("answers"), default="").strip()

        out.append(_mk_record(
            _id=f"web_questions_{i}",
            task_type="open_qa",
            messages=_mk_messages(user, answer),
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_wiki_qa(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("wiki_qa")
    ds = datasets.load_dataset("wiki_qa", split=split)
    print(f"split={split} rows={len(ds)} (before label filter)")

    out: list[dict[str, Any]] = []
    kept = 0
    for i, sample in enumerate(ds):
        if sample.get("label") != 1:
            continue
        kept += 1
        if max_examples is not None and len(out) >= max_examples:
            break

        user = _format_qa_question(sample["question"], rng)
        answer = (sample["answer"] or "").strip()
        out.append(_mk_record(
            _id=f"wiki_qa_{i}",
            task_type="open_qa",
            messages=_mk_messages(user, answer),
        ))

    print(f"kept_label1={kept} prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_yahoo_answers_qa(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("yahoo_answers_qa")
    ds = datasets.load_dataset("yahoo_answers_qa", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        user = _format_qa_question(sample["question"], rng)
        answer = (sample.get("answer") or "").strip()
        out.append(_mk_record(
            _id=f"yahoo_answers_qa_{i}",
            task_type="open_qa",
            messages=_mk_messages(user, answer),
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_freebase_qa(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("freebase_qa")
    ds = datasets.load_dataset("freebase_qa", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        user = _format_qa_question(sample.get("RawQuestion", ""), rng)

        ans = ""
        try:
            # sample["Parses"]["Answers"][0]["AnswersName"][0][0]
            ans = sample["Parses"]["Answers"][0]["AnswersName"][0][0]
        except Exception:
            ans = ""

        out.append(_mk_record(
            _id=f"freebase_qa_{i}",
            task_type="open_qa",
            messages=_mk_messages(user, str(ans).strip()),
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_ms_marco(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("ms_marco:v2.1")
    ds = datasets.load_dataset("ms_marco", "v2.1", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        q = (sample.get("query") or "").lstrip(")").strip()
        user = _format_qa_question(q, rng)
        answer = _safe_first(sample.get("answers"), default="").strip()

        out.append(_mk_record(
            _id=f"ms_marco_{i}",
            task_type="open_qa",
            messages=_mk_messages(user, answer),
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_coqa(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("coqa")
    ds = datasets.load_dataset("coqa", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        qs = sample["questions"]
        ans = sample["answers"]["input_text"]
        story = sample.get("story", "")

        if len(qs) != len(ans):
            continue

        messages: list[dict[str, str]] = []
        for turn_idx, (q, a) in enumerate(zip(qs, ans)):
            q = (q or "").strip()
            if turn_idx == 0:
                q = _format_qa_question(q, rng)
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": (a or "").strip()})

        out.append(_mk_record(
            _id=f"coqa_{i}",
            task_type="close_qa",
            messages=messages,
            background=(story or "").strip(),
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_drop(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("drop")
    ds = datasets.load_dataset("drop", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        user = _format_qa_question(sample.get("question", ""), rng)
        spans = sample.get("answers_spans", {}).get("spans", [])
        answer = _safe_first(spans, default="").strip()
        passage = (sample.get("passage") or "").strip()

        out.append(_mk_record(
            _id=f"drop_{i}",
            task_type="close_qa",
            messages=_mk_messages(user, answer),
            background=passage,
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_narrativeqa(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("narrativeqa")
    ds = datasets.load_dataset("narrativeqa", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        question = sample.get("question", {}).get("text", "")
        answer = _safe_first(sample.get("answers"), default={})
        answer_text = (answer.get("text") if isinstance(answer, dict) else str(answer)).strip()
        user = _format_qa_question(question, rng)

        background = (sample.get("document", {}) or {}).get("summary", {}) or {}
        bg_text = (background.get("text", "") if isinstance(background, dict) else str(background)).strip()

        out.append(_mk_record(
            _id=f"narrativeqa_{i}",
            task_type="close_qa",
            messages=_mk_messages(user, answer_text),
            background=bg_text,
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_triviaqa_local(*, path: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header(f"triviaqa_local: {path}")
    rows = _read_jsonl(path)
    if max_examples is not None:
        rows = rows[:max_examples]
    print(f"rows={len(rows)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(rows):
        question = sample.get("question", "")
        answer = _safe_first(sample.get("answer"), default="").strip()
        user = _format_qa_question(question, rng)

        out.append(_mk_record(
            _id=f"triviaqa_{i}",
            task_type="open_qa",
            messages=_mk_messages(user, answer),
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_fm2_local(*, path: str, max_examples: int | None) -> list[dict[str, Any]]:
    _print_header(f"fm2_local: {path}")
    rows = _read_jsonl(path)
    if max_examples is not None:
        rows = rows[:max_examples]
    print(f"rows={len(rows)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(rows):
        claims = sample.get("question", "")
        ans = sample.get("answer", "")
        label = "True" if isinstance(ans, str) and ("supports" in ans) else "False"

        out.append(_mk_record(
            _id=f"fm2_{i}",
            task_type="fact_checking",
            messages=_mk_messages(_format_fact_prompt(claims), label),
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_nq_open(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("nq_open")
    ds = datasets.load_dataset("nq_open", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        question = sample.get("question", "")
        answer = _safe_first(sample.get("answer"), default="").strip()
        user = _format_qa_question(question, rng)

        out.append(_mk_record(
            _id=f"nq_{i}",
            task_type="open_qa",
            messages=_mk_messages(user, answer),
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_pwc_local(*, path: str, max_examples: int | None) -> list[dict[str, Any]]:
    _print_header(f"pwc_local: {path}")
    rows = _read_jsonl(path)
    if max_examples is not None:
        rows = rows[:max_examples]
    print(f"rows={len(rows)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(rows):
        out.append(_mk_record(
            _id=f"pwc_{i}",
            task_type="close_qa",
            messages=_mk_messages(sample.get("prompt", ""), sample.get("answer", "")),
            background=(sample.get("input") or "").strip(),
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_dialogsum(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("dialogsum")
    ds = datasets.load_dataset("knkarthick/dialogsum", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        user = _format_sum_prompt(rng)
        answer = (sample.get("summary") or "").strip()
        background = (sample.get("dialogue") or "").strip()

        out.append(_mk_record(
            _id=f"dialogsum_{i}",
            task_type="summarization",
            messages=_mk_messages(user, answer),
            background=background,
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_samsum(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("samsum")
    ds = datasets.load_dataset("samsum", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    skipped_empty = 0
    for i, sample in enumerate(ds):
        background = (sample.get("dialogue") or "").replace("\r\n", "\n").strip()
        if not background:
            skipped_empty += 1
            continue

        user = _format_sum_prompt(rng)
        answer = (sample.get("summary") or "").strip()
        out.append(_mk_record(
            _id=f"samsum_{i}",
            task_type="summarization",
            messages=_mk_messages(user, answer),
            background=background,
        ))

    print(f"prepared={len(out)} skipped_empty={skipped_empty} last_id={out[-1]['id'] if out else None}")
    return out


def load_cnn_dailymail(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("cnn_dailymail:3.0.0")
    ds = datasets.load_dataset("cnn_dailymail", "3.0.0", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        user = _format_sum_prompt(rng)
        answer = (sample.get("highlights") or "").strip()
        background = (sample.get("article") or "").strip()

        out.append(_mk_record(
            _id=f"cnn_dailymail_{i}",
            task_type="summarization",
            messages=_mk_messages(user, answer),
            background=background,
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_squad_v2(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("squad_v2")
    ds = datasets.load_dataset("squad_v2", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        question = sample.get("question", "")
        answers = sample.get("answers", {}).get("text", []) or []
        answer = answers[0] if len(answers) > 0 else "I don't know."
        user = _format_qa_question(question, rng)
        background = (sample.get("context") or "").strip()

        out.append(_mk_record(
            _id=f"squad_v2_{i}",
            task_type="close_qa",
            messages=_mk_messages(user, (answer or "").strip()),
            background=background,
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_quail(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("quail")
    ds = datasets.load_dataset("quail", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    labels = ["A", "B", "C", "D"]
    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        q = (sample.get("question") or "").strip() + "\n"
        answers = sample.get("answers") or []
        for j, a in enumerate(answers[:4]):
            q += f"{labels[j]}. {a}\n"

        correct_id = int(sample.get("correct_answer_id", 0))
        answer = labels[correct_id] if 0 <= correct_id < len(labels) else labels[0]

        user = _format_qa_question(q, rng)
        background = (sample.get("context") or "").strip()

        out.append(_mk_record(
            _id=f"quail_{i}",
            task_type="close_qa",
            messages=_mk_messages(user, answer),
            background=background,
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


def load_pubmed_qa(*, split: str, max_examples: int | None, rng: random.Random) -> list[dict[str, Any]]:
    _print_header("pubmed_qa:pqa_labeled")
    ds = datasets.load_dataset("pubmed_qa", "pqa_labeled", split=split)
    ds = _maybe_select(ds, max_examples)
    print(f"split={split} rows={len(ds)}")

    out: list[dict[str, Any]] = []
    for i, sample in enumerate(ds):
        question = sample.get("question", "")
        user = _format_qa_question(question, rng)

        long_answer = (sample.get("long_answer") or "").strip()
        final_decision = (sample.get("final_decision") or "").strip()
        answer = f"{long_answer} So the final answer is: {final_decision}".strip()

        ctx = sample.get("context", {}) or {}
        contexts = ctx.get("contexts", []) if isinstance(ctx, dict) else []
        background = "\n".join([str(x).strip() for x in contexts if str(x).strip()])

        out.append(_mk_record(
            _id=f"pubmed_qa_{i}",
            task_type="close_qa",
            messages=_mk_messages(user, answer),
            background=background,
        ))

    print(f"prepared={len(out)} last_id={out[-1]['id'] if out else None}")
    return out


# ----------------------------
# Main
# ----------------------------

@dataclass(frozen=True)
class Paths:
    triviaqa_jsonl: str | None
    fm2_jsonl: str | None
    pwc_jsonl: str | None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", type=str, required=True, help="Where to write merged JSONL.")
    ap.add_argument("--out_hf_dir", type=str, default=None, help="Optional Dataset.save_to_disk() path.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_per_dataset", type=int, default=None, help="Cap examples per dataset (after filtering).")
    ap.add_argument("--shuffle", action="store_true")

    # Local files (optional)
    ap.add_argument("--triviaqa_jsonl", type=str, default=None)
    ap.add_argument("--fm2_jsonl", type=str, default=None)
    ap.add_argument("--pwc_jsonl", type=str, default=None)

    # Include/exclude toggles (simple)
    ap.add_argument("--skip", type=str, nargs="*", default=[], help="Dataset keys to skip.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    paths = Paths(triviaqa_jsonl=args.triviaqa_jsonl, fm2_jsonl=args.fm2_jsonl, pwc_jsonl=args.pwc_jsonl)

    # Registry: (key, loader)
    loaders: list[tuple[str, Callable[[], list[dict[str, Any]]]]] = [
        ("commonsense_qa", lambda: load_commonsense_qa(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("web_questions", lambda: load_web_questions(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("wiki_qa", lambda: load_wiki_qa(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("yahoo_answers_qa", lambda: load_yahoo_answers_qa(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("freebase_qa", lambda: load_freebase_qa(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("ms_marco", lambda: load_ms_marco(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("coqa", lambda: load_coqa(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("drop", lambda: load_drop(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("narrativeqa", lambda: load_narrativeqa(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("nq_open", lambda: load_nq_open(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("dialogsum", lambda: load_dialogsum(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("samsum", lambda: load_samsum(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("cnn_dailymail", lambda: load_cnn_dailymail(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("squad_v2", lambda: load_squad_v2(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("quail", lambda: load_quail(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
        ("pubmed_qa", lambda: load_pubmed_qa(split=args.split, max_examples=args.max_per_dataset, rng=rng)),
    ]

    if paths.triviaqa_jsonl:
        loaders.append(("triviaqa_local", lambda: load_triviaqa_local(path=paths.triviaqa_jsonl, max_examples=args.max_per_dataset, rng=rng)))
    if paths.fm2_jsonl:
        loaders.append(("fm2_local", lambda: load_fm2_local(path=paths.fm2_jsonl, max_examples=args.max_per_dataset)))
    if paths.pwc_jsonl:
        loaders.append(("pwc_local", lambda: load_pwc_local(path=paths.pwc_jsonl, max_examples=args.max_per_dataset)))

    skip = set(args.skip or [])
    merged: list[dict[str, Any]] = []
    per_dataset_counts: dict[str, int] = {}

    for key, fn in loaders:
        if key in skip:
            print(f"\n=== {key} (skipped) ===")
            continue
        try:
            rows = fn()
            per_dataset_counts[key] = len(rows)
            merged.extend(rows)
        except Exception as e:
            print(f"Failed to download {key}")
            print("ERROR: ", e)

    # Deduplicate IDs (keep first)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    dup = 0
    for r in merged:
        rid = r["id"]
        if rid in seen:
            dup += 1
            continue
        seen.add(rid)
        deduped.append(r)

    if args.shuffle:
        rng.shuffle(deduped)

    print("\n=== Summary ===")
    for k in sorted(per_dataset_counts.keys()):
        print(f"{k:16s} {per_dataset_counts[k]:8d}")
    print(f"total_before_dedup={len(merged)} total_after_dedup={len(deduped)} duplicates_dropped={dup}")

    n = _write_jsonl(deduped, args.out_jsonl)
    print(f"wrote_jsonl={n} -> {args.out_jsonl}")

    if args.out_hf_dir:
        os.makedirs(args.out_hf_dir, exist_ok=True)
        ds = datasets.Dataset.from_list(deduped)
        ds.save_to_disk(args.out_hf_dir)
        print(f"saved_hf_dataset -> {args.out_hf_dir}")


if __name__ == "__main__":
    main()
