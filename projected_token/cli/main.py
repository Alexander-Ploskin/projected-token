from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import click

from projected_token.config import load_yaml
from projected_token.evaluation import evaluate_paraphrase, evaluate_qa
from projected_token.generation import run_generation
from projected_token.pipeline import run_all as run_all_experiments
from projected_token.pipeline import run_experiment
from projected_token.retrieval.pipeline import build_index, evaluate_retrieval
from projected_token.training.recipe_runner import run_training


def _judge_config(base_url: str | None, api_key: str | None, model: str | None, config: str | None = None) -> dict[str, Any] | None:
    data = load_yaml(config).get("evaluation", {}) if config else {}
    resolved = {
        "base_url": base_url or data.get("base_url") or data.get("llm_base_url"),
        "api_key": api_key or data.get("api_key") or data.get("llm_api_key"),
        "model": model or data.get("model") or data.get("llm_model"),
        "temperature": data.get("temperature", 0.0),
    }
    return resolved if all([resolved["base_url"], resolved["api_key"], resolved["model"]]) else None


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Projected Token experiments CLI."""


@cli.command()
@click.option("--task", type=click.Choice(["paraphrase", "qa"]), required=True)
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option("--text-col", default="s_wiki_content")
@click.option("--question-col", default="question")
@click.option("--output-col", default=None)
@click.option("--batch-size", type=int, default=None)
def generate(task: str, config_path: str, input_path: str, output_path: str, text_col: str, question_col: str, output_col: str | None, batch_size: int | None) -> None:
    """Generate paraphrases or QA answers."""
    run_generation(config_path=config_path, input_path=input_path, output_path=output_path, task=task, text_col=text_col, question_col=question_col, output_col=output_col, batch_size=batch_size)


@cli.command()
@click.option("--task", type=click.Choice(["paraphrase", "qa"]), required=True)
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option("--config", "config_path", default=None, type=click.Path(exists=True))
@click.option("--base-url", default=None)
@click.option("--api-key", default=None)
@click.option("--model", default=None)
@click.option("--original-col", default="s_wiki_content")
@click.option("--candidate-col", default="paraphrase")
@click.option("--question-col", default="question")
@click.option("--answer-col", default="obj")
@click.option("--prediction-col", default="answer")
@click.option("--batch-size", default=10)
def evaluate(task: str, input_path: str, output_path: str, config_path: str | None, base_url: str | None, api_key: str | None, model: str | None, original_col: str, candidate_col: str, question_col: str, answer_col: str, prediction_col: str, batch_size: int) -> None:
    """Evaluate generated outputs."""
    judge = _judge_config(base_url, api_key, model, config_path)
    if task == "qa":
        evaluate_qa(input_path=input_path, output_path=output_path, judge_config=judge, question_col=question_col, answer_col=answer_col, prediction_col=prediction_col, batch_size=batch_size)
    else:
        metric_configs = load_yaml(config_path).get("metrics", []) if config_path else []
        evaluate_paraphrase(input_path=input_path, output_path=output_path, original_col=original_col, candidate_col=candidate_col, judge_config=judge, metric_configs=metric_configs, batch_size=batch_size)


@cli.command(name="run")
@click.option("--task", type=click.Choice(["paraphrase", "qa"]), required=True)
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--text-col", default="s_wiki_content")
@click.option("--question-col", default="question")
@click.option("--answer-col", default="obj")
@click.option("--batch-size", type=int, default=None)
@click.option("--judge-base-url", default=None)
@click.option("--judge-api-key", default=None)
@click.option("--judge-model", default=None)
def run_cmd(task: str, config_path: str, input_path: str, output_dir: str, text_col: str, question_col: str, answer_col: str, batch_size: int | None, judge_base_url: str | None, judge_api_key: str | None, judge_model: str | None) -> None:
    """Run generation followed by evaluation."""
    run_experiment(task=task, config_path=config_path, input_path=input_path, output_dir=output_dir, text_col=text_col, question_col=question_col, answer_col=answer_col, batch_size=batch_size, judge_base_url=judge_base_url, judge_api_key=judge_api_key, judge_model=judge_model)


@cli.command(name="run-all")
@click.option("--task", type=click.Choice(["paraphrase", "qa"]), required=True)
@click.option("--configs-dir", required=True, type=click.Path(exists=True))
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--config-filter", default=None)
@click.option("--batch-size", type=int, default=None)
def run_all_cmd(task: str, configs_dir: str, input_path: str, output_dir: str, config_filter: str | None, batch_size: int | None) -> None:
    """Run all matching experiment configs."""
    run_all_experiments(task=task, configs_dir=configs_dir, input_path=input_path, output_dir=output_dir, config_filter=config_filter, batch_size=batch_size)


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--epochs", type=int, default=None)
def train(config_path: str, epochs: int | None) -> None:
    """Train a projector using a YAML recipe."""
    run_training(config_path, epochs=epochs)


@cli.group()
def retrieval() -> None:
    """Build and evaluate retrieval indexes."""


@retrieval.command(name="build-index")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def retrieval_build_index(config_path: str) -> None:
    build_index(config_path)


@retrieval.command(name="evaluate")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def retrieval_evaluate(config_path: str) -> None:
    evaluate_retrieval(config_path)


@retrieval.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, name="index-kilt-sfr")
@click.pass_context
def retrieval_index_kilt_sfr(ctx: click.Context) -> None:
    """Index s-nlp/kilt with Salesforce/SFR-Embedding-Mistral."""
    _run_recipe_module("projected_token.retrieval.recipes.index_kilt_sfr", tuple(ctx.args))


@retrieval.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, name="index-kilt-bm25")
@click.pass_context
def retrieval_index_kilt_bm25(ctx: click.Context) -> None:
    """Index s-nlp/kilt with BM25 (bm25s)."""
    _run_recipe_module("projected_token.retrieval.recipes.index_kilt_bm25", tuple(ctx.args))


@cli.group()
def data() -> None:
    """Prepare datasets and teacher embeddings."""


@data.command(name="prepare-popqa")
@click.option("--output-dir", default="data/eval/popqa")
@click.option("--workers", default=8)
@click.option("--batch-size", default=50)
def data_prepare_popqa(output_dir: str, workers: int, batch_size: int) -> None:
    from projected_token.data.eval.popqa import PopQADatasetPreparer
    PopQADatasetPreparer(output_dir=output_dir, max_workers=workers, batch_size=batch_size).run()


def _run_recipe_module(module_name: str, args: tuple[str, ...]) -> None:
    module = importlib.import_module(module_name)
    old_argv = sys.argv
    try:
        sys.argv = [module_name, *args]
        module.main()
    finally:
        sys.argv = old_argv


@data.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, name="prepare-msmarco")
@click.pass_context
def data_prepare_msmarco(ctx: click.Context) -> None:
    _run_recipe_module("projected_token.data.recipes.prepare_msmarco", tuple(ctx.args))


@data.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, name="prepare-msmarco-v2")
@click.pass_context
def data_prepare_msmarco_v2(ctx: click.Context) -> None:
    _run_recipe_module("projected_token.data.recipes.prepare_msmarco_v2", tuple(ctx.args))


@data.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, name="mixed-dataset")
@click.pass_context
def data_mixed_dataset(ctx: click.Context) -> None:
    _run_recipe_module("projected_token.data.recipes.generate_mixed_dataset", tuple(ctx.args))


@data.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, name="hard-negatives")
@click.pass_context
def data_hard_negatives(ctx: click.Context) -> None:
    _run_recipe_module("projected_token.data.recipes.generate_hard_negatives", tuple(ctx.args))


@data.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, name="teacher-embeddings")
@click.pass_context
def data_teacher_embeddings(ctx: click.Context) -> None:
    _run_recipe_module("projected_token.data.recipes.generate_teacher_embeddings", tuple(ctx.args))
