from __future__ import annotations

from pathlib import Path
from typing import Any

from projected_token.config import load_yaml
from projected_token.evaluation import evaluate_paraphrase, evaluate_qa
from projected_token.generation import run_generation


def _judge_config(config: dict[str, Any], *, base_url: str | None, api_key: str | None, model: str | None) -> dict[str, Any] | None:
    evaluation = config.get("evaluation", {})
    resolved = {
        "base_url": base_url or evaluation.get("base_url") or evaluation.get("llm_base_url"),
        "api_key": api_key or evaluation.get("api_key") or evaluation.get("llm_api_key"),
        "model": model or evaluation.get("model") or evaluation.get("llm_model"),
        "temperature": evaluation.get("temperature", 0.0),
    }
    if not all([resolved["base_url"], resolved["api_key"], resolved["model"]]):
        return None
    return resolved


def run_experiment(
    *,
    task: str,
    config_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    text_col: str = "s_wiki_content",
    question_col: str = "question",
    answer_col: str = "obj",
    batch_size: int | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(config_path)
    generated_path = output_dir / f"{task}_generations.jsonl"
    metrics_path = output_dir / f"{task}_metrics.json"
    run_generation(
        config_path=config_path,
        input_path=input_path,
        output_path=generated_path,
        task=task,
        text_col=text_col,
        question_col=question_col,
        batch_size=batch_size,
    )
    judge = _judge_config(config, base_url=judge_base_url, api_key=judge_api_key, model=judge_model)
    if task == "qa":
        return evaluate_qa(input_path=generated_path, output_path=metrics_path, judge_config=judge, question_col=question_col, answer_col=answer_col)
    return evaluate_paraphrase(input_path=generated_path, output_path=metrics_path, judge_config=judge, original_col=text_col)


def run_all(
    *,
    task: str,
    configs_dir: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    config_filter: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for config_path in sorted(Path(configs_dir).glob("*.yaml")):
        if config_filter and config_filter not in config_path.name:
            continue
        run_dir = Path(output_dir) / config_path.stem
        results[config_path.stem] = run_experiment(task=task, config_path=config_path, input_path=input_path, output_dir=run_dir, **kwargs)
    return results
