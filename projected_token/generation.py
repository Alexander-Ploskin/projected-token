from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from tqdm import tqdm

from projected_token.config import instantiate, load_yaml, validate_experiment_config
from projected_token.io import write_jsonl
from projected_token.logging_utils import log_step, log_warning


def _prompt_template(model: Any, experiment: dict[str, Any], task: str) -> str:
    configured = experiment.get("prompt_template")
    if configured:
        return configured
    if task == "qa":
        return getattr(model, "QA_DEFAULT_PROMPT_TEMPLATE", None) or getattr(model, "default_prompt_template", None)
    return getattr(model, "PARAPHRASE_DEFAULT_PROMPT_TEMPLATE", None) or getattr(model, "default_prompt_template", None)


def _call_generate_batch(model: Any, docs: list[str], prompt_template: str, generation: dict[str, Any], task: str, questions: list[str] | None) -> list[str]:
    if task == "qa":
        signature = inspect.signature(model.generate_batch)
        if "questions" in signature.parameters:
            return model.generate_batch(docs, prompt_template, generation, questions=questions)
    return model.generate_batch(docs, prompt_template, generation)


def run_generation(
    *,
    config_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    task: str,
    text_col: str = "s_wiki_content",
    question_col: str = "question",
    output_col: str | None = None,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    config = validate_experiment_config(load_yaml(config_path)).model_dump(by_alias=True)
    dataset = instantiate(config["dataset"], kind="dataset").load(str(input_path))
    model = instantiate(config["model"], kind="model")
    generation = config.get("generation", {})
    experiment = config.get("experiment", {})
    sample_count = int(experiment.get("sample_count", 1))
    output_col = output_col or ("answer" if task == "qa" else "paraphrase")
    prompt_template = _prompt_template(model, experiment, task)
    if not prompt_template:
        raise ValueError(f"No prompt template configured for task '{task}'")

    records = list(dataset)
    supports_batch = hasattr(model, "generate_batch") and callable(model.generate_batch)
    if not supports_batch:
        batch_size = None
    elif batch_size is None:
        batch_size = 1

    outputs: list[dict[str, Any]] = []
    log_step(f"Generating {task} outputs for {len(records)} records")
    for start in tqdm(range(0, len(records), batch_size or 1), desc=f"generate-{task}"):
        batch = records[start:start + (batch_size or 1)]
        expanded: list[dict[str, Any]] = []
        docs: list[str] = []
        questions: list[str] = []
        for item in batch:
            doc = item.get(text_col, "") or ""
            question = item.get(question_col, "") or ""
            for _ in range(sample_count):
                expanded.append(item)
                docs.append(doc)
                if task == "qa":
                    questions.append(question)
        try:
            if supports_batch:
                generated = _call_generate_batch(model, docs, prompt_template, generation, task, questions if task == "qa" else None)
            else:
                generated = [model(doc, prompt_template, generation) for doc in docs]
        except Exception as exc:
            log_warning(f"Batch failed at offset {start}: {exc}. Falling back to single-item generation.")
            generated = []
            for idx, doc in enumerate(docs):
                if task == "qa":
                    generated.append(model.generate_batch([doc], prompt_template, generation, questions=[questions[idx]])[0])
                else:
                    generated.append(model(doc, prompt_template, generation))
        for item, generated_text in zip(expanded, generated):
            row = dict(item)
            row[output_col] = generated_text
            outputs.append(row)
    write_jsonl(output_path, outputs)
    log_step(f"Wrote {len(outputs)} rows to {output_path}")
    return outputs
