from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from projected_token.io import load_jsonl, write_json, write_jsonl
from projected_token.qa.eval import evaluate_predictions_jsonl

PAPER_RAG_SYSTEM = (
    "You are a helpful assistant. Your task is to extract relevant information "
    "from provided documents and to answer questions as briefly as possible."
)
NO_CONTEXT_SYSTEM = (
    "You are a helpful assistant. Answer using your own knowledge only. "
    "Do not refer to external documents or sources you were not given in this chat. "
    "Be as brief and factual as possible."
)
PAPER_DOC_SEP = "SEP"


def _dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _answers_from_value(value: Any) -> list[str]:
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _qid_from_row(row: dict[str, Any], fallback: int) -> str:
    qid = row.get("qid", row.get("query_index"))
    return str(qid if qid is not None else fallback)


def _question_from_row(row: dict[str, Any]) -> str:
    return str(row.get("question", row.get("query", "")) or "")


def _doc_texts_from_row(row: dict[str, Any], top_k: int | None) -> list[str]:
    docs = row.get("docs", [])
    if not isinstance(docs, list):
        return []
    texts: list[str] = []
    for doc in docs[:top_k]:
        if isinstance(doc, dict):
            text = str(doc.get("text", "") or "").strip()
        else:
            text = str(doc or "").strip()
        if text:
            texts.append(text)
    return texts


def _load_prompt_template(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


class SimpleQAGenerator:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str,
        torch_dtype: str,
        use_chat_template: bool,
    ) -> None:
        self.device = torch.device(device)
        dtype = _dtype(torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat_template = use_chat_template

    def _chat_or_plain(self, messages: list[dict[str, str]], plain: str) -> str:
        if self.use_chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        return plain

    def _rag_prompt(self, question: str, context: str, prompt_template: str | None) -> str:
        if prompt_template is not None:
            raw = prompt_template.format(question=question.strip(), context=context.strip())
            return self._chat_or_plain([{"role": "user", "content": raw}], raw)
        user = f"Background:\n{context.strip()}\nQuestion: {question.strip()}"
        return self._chat_or_plain(
            [{"role": "system", "content": PAPER_RAG_SYSTEM}, {"role": "user", "content": user}],
            f"{PAPER_RAG_SYSTEM}\n\n{user}",
        )

    def _no_context_prompt(self, question: str, prompt_template: str | None) -> str:
        if prompt_template is not None:
            if "{context}" in prompt_template:
                raise ValueError("No-context prompt template must not include {context}.")
            raw = prompt_template.format(question=question.strip())
            return self._chat_or_plain([{"role": "user", "content": raw}], raw)
        user = f"Question: {question.strip()}"
        return self._chat_or_plain(
            [{"role": "system", "content": NO_CONTEXT_SYSTEM}, {"role": "user", "content": user}],
            f"{NO_CONTEXT_SYSTEM}\n\n{user}",
        )

    def _build_context(
        self,
        *,
        question: str,
        texts: list[str],
        prompt_template: str | None,
        max_input_tokens: int,
    ) -> str:
        working = list(texts)
        while working:
            context = PAPER_DOC_SEP.join(working) if prompt_template is None else "\n\n".join(
                f"[{idx + 1}] {text}" for idx, text in enumerate(working)
            )
            prompt = self._rag_prompt(question, context, prompt_template)
            if len(self.tokenizer.encode(prompt, add_special_tokens=True)) <= max_input_tokens:
                return context
            working.pop()
        return ""

    def generate(
        self,
        *,
        method: str,
        questions: list[str],
        texts_per_question: list[list[str]],
        prompt_template: str | None,
        max_input_tokens: int,
        max_length: int,
        generation_args: dict[str, Any],
    ) -> list[str]:
        prompts: list[str] = []
        for question, texts in zip(questions, texts_per_question):
            if method == "llm_nocontext":
                prompts.append(self._no_context_prompt(question, prompt_template))
            else:
                context = self._build_context(
                    question=question,
                    texts=texts,
                    prompt_template=prompt_template,
                    max_input_tokens=max_input_tokens,
                )
                prompts.append(self._rag_prompt(question, context, prompt_template))

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        gen_kwargs = dict(generation_args)
        if not bool(gen_kwargs.get("do_sample", False)):
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                **gen_kwargs,
            )

        predictions: list[str] = []
        for output, prompt_ids in zip(outputs, input_ids):
            answer_ids = output[prompt_ids.shape[0] :]
            predictions.append(self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip())
        return predictions


class OscarQAGenerator:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str,
        torch_dtype: str,
        trust_remote_code: bool,
    ) -> None:
        dtype = _dtype(torch_dtype)
        device_map = device if device != "cpu" else "cpu"
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            low_cpu_mem_usage=True,
        ).eval()

    def generate(
        self,
        *,
        questions: list[str],
        texts_per_question: list[list[str]],
        max_new_tokens: int,
        query_dependent: bool,
    ) -> list[str]:
        predictions = [""] * len(questions)
        valid_items: list[tuple[int, str, list[str]]] = []
        for idx, (question, texts) in enumerate(zip(questions, texts_per_question)):
            clean_question = question.strip()
            clean_docs = [text.strip() for text in texts if text.strip()]
            if clean_question and clean_docs:
                valid_items.append((idx, clean_question, clean_docs))
        if not valid_items:
            return predictions

        if hasattr(self.model, "generate_from_text"):
            max_docs = max(len(docs) for _, _, docs in valid_items)
            batch_questions = [question for _, question, _ in valid_items]
            batch_contexts = [docs + [""] * (max_docs - len(docs)) for _, _, docs in valid_items]
            generate_from_text = self.model.generate_from_text
            params = inspect.signature(generate_from_text).parameters
            kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
            if "query_dependent" in params:
                kwargs["query_dependent"] = query_dependent
            with torch.inference_mode():
                if "documents" in params:
                    outputs = generate_from_text(
                        questions=batch_questions,
                        documents=batch_contexts,
                        **kwargs,
                    )
                elif "contexts" in params:
                    outputs = generate_from_text(
                        contexts=batch_contexts,
                        questions=batch_questions,
                        **kwargs,
                    )
                else:
                    outputs = generate_from_text(batch_contexts, batch_questions, **kwargs)
            for (idx, _, _), output in zip(valid_items, outputs):
                predictions[idx] = str(output).strip()
            return predictions

        for idx, question, clean_docs in valid_items:
            compression_questions = [question] * len(clean_docs) if query_dependent else None
            with torch.inference_mode():
                embeddings = self.model.compress_documents(
                    documents=clean_docs,
                    questions=compression_questions,
                )
                output = self.model.generate_from_compressed_documents_and_questions(
                    questions=[question],
                    compressed_documents=embeddings,
                    max_new_tokens=max_new_tokens,
                )
            predictions[idx] = str(output[0] if isinstance(output, list) and output else output).strip()
        return predictions


def _load_done_qids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("qid")
            if qid is not None:
                done.add(str(qid))
    return done


def run(args: argparse.Namespace) -> dict[str, Any]:
    topk_path = Path(args.topk_jsonl)
    out_pred_path = Path(args.out_pred)
    rows = load_jsonl(topk_path)
    if args.max_rows is not None:
        rows = rows[: max(0, int(args.max_rows))]

    prompt_template = _load_prompt_template(args.prompt_template)
    done_qids = _load_done_qids(out_pred_path) if args.resume else set()

    if args.method in {"rag", "llm_nocontext"}:
        generator: SimpleQAGenerator | OscarQAGenerator = SimpleQAGenerator(
            model_name_or_path=args.llm_model,
            device=args.device,
            torch_dtype=args.llm_torch_dtype,
            use_chat_template=not args.no_chat_template,
        )
        model_id = args.llm_model
    else:
        generator = OscarQAGenerator(
            model_name_or_path=args.oscar_model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            trust_remote_code=not args.no_trust_remote_code,
        )
        model_id = args.oscar_model

    out_pred_path.parent.mkdir(parents=True, exist_ok=True)
    first_write = not args.resume or not out_pred_path.exists() or out_pred_path.stat().st_size == 0
    stats = {"n_written": 0, "n_skipped_resume": 0}

    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
    }
    if args.top_p is not None:
        generation_args["top_p"] = args.top_p

    batch: list[tuple[int, dict[str, Any]]] = []

    def flush() -> None:
        nonlocal batch, first_write
        if not batch:
            return
        questions = [_question_from_row(row) for _, row in batch]
        texts_per_question = [_doc_texts_from_row(row, args.top_k) for _, row in batch]
        if args.method == "oscar":
            assert isinstance(generator, OscarQAGenerator)
            predictions = generator.generate(
                questions=questions,
                texts_per_question=texts_per_question,
                max_new_tokens=args.max_new_tokens,
                query_dependent=args.oscar_query_dependent,
            )
        else:
            assert isinstance(generator, SimpleQAGenerator)
            predictions = generator.generate(
                method=args.method,
                questions=questions,
                texts_per_question=texts_per_question,
                prompt_template=prompt_template,
                max_input_tokens=args.max_input_tokens,
                max_length=args.max_length,
                generation_args=generation_args,
            )

        mode = "w" if first_write else "a"
        first_write = False
        records: list[dict[str, Any]] = []
        for (fallback_idx, row), question, texts, prediction in zip(
            batch, questions, texts_per_question, predictions
        ):
            records.append(
                {
                    "qid": _qid_from_row(row, fallback_idx),
                    "question": question,
                    "prediction": prediction,
                    "method": args.method,
                    "model_id": model_id,
                    "topk_source": str(topk_path),
                    "answers": _answers_from_value(row.get("answers", row.get("targets"))),
                    **({"used_doc_ids": [str(doc.get("doc_id", "")) for doc in row.get("docs", [])[: args.top_k]]} if args.include_retrieval else {}),
                    **({"used_texts": texts} if args.include_retrieval_text else {}),
                }
            )
        with out_pred_path.open(mode, encoding="utf-8") as fp:
            for record in records:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats["n_written"] += len(records)
        batch = []

    for idx, row in enumerate(tqdm(rows, desc=f"qa-generate-{args.method}", unit="q", dynamic_ncols=True)):
        qid = _qid_from_row(row, idx)
        if qid in done_qids:
            stats["n_skipped_resume"] += 1
            continue
        batch.append((idx, row))
        if len(batch) >= args.batch_size:
            flush()
    flush()

    meta = {
        "cmd": "qa e2e-from-topk",
        "method": args.method,
        "model_id": model_id,
        "topk_jsonl": str(topk_path.resolve()),
        "out_pred": str(out_pred_path.resolve()),
        "config": {
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "max_rows": args.max_rows,
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "max_length": args.max_length,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "oscar_query_dependent": args.oscar_query_dependent,
        },
        "stats": stats,
    }
    meta_path = out_pred_path.with_name(f"{out_pred_path.stem}_meta.json")
    write_json(meta_path, meta)

    metrics = evaluate_predictions_jsonl(
        pred_path=out_pred_path,
        output_path=args.out_metrics,
        gold_jsonl=args.gold_jsonl,
        gold_hf=args.gold_hf,
        hf_split=args.hf_split,
        max_rows=args.max_rows,
        hf_cache_dir=args.hf_cache_dir,
        hf_token=args.hf_token,
        write_errors_path=args.write_errors,
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and score QA predictions from retrieval top-k JSONL.")
    parser.add_argument("--topk-jsonl", required=True, help="Saved *_topk.jsonl from retrieval eval-kilt-openqa.")
    parser.add_argument("--method", choices=("llm_nocontext", "rag", "oscar"), required=True)
    parser.add_argument("--out-pred", required=True)
    parser.add_argument("--out-metrics", required=True)
    parser.add_argument("--gold-jsonl", default=None, help="Optional gold JSONL. Defaults to inline answers in top-k rows.")
    parser.add_argument("--gold-hf", choices=("popqa", "hotpotqa", "hotpotqa_distractor", "hotpotqa_fullwiki"), default=None)
    parser.add_argument("--hf-split", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--write-errors", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None, help="Limit number of retrieved docs used for QA.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--include-retrieval", action="store_true")
    parser.add_argument("--include-retrieval-text", action="store_true")

    parser.add_argument("--llm-model", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--llm-torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--prompt-template", default=None, help="RAG: {question}/{context}; no-context: {question}. Use @path to load from file.")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=None)

    parser.add_argument("--oscar-model", default="naver/oscar-qwen2-7B")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument(
        "--oscar-query-dependent",
        dest="oscar_query_dependent",
        action="store_true",
        default=False,
        help="Compress OSCAR documents conditioned on the question.",
    )
    parser.add_argument(
        "--no-oscar-query-dependent",
        dest="oscar_query_dependent",
        action="store_false",
        help="Use faster query-independent OSCAR document compression.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
