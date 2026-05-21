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
from projected_token.retrieval.beir import evaluate_beir
from projected_token.training.matrix_runner import run_matrix
from projected_token.training.recipe_runner import run_training
from projected_token.training.final_report import build_final_report
from projected_token.training.oscar_retrieval_roadmap import run_roadmap
from projected_token.analysis.aggregate_beir3_results import aggregate_beir3_results
from projected_token.analysis.mem_latent_diagnostics import DEFAULT_POOLERS, run_mem_latent_diagnostics


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
    result = run_training(config_path, epochs=epochs)
    if isinstance(result, dict) and result.get("run_root"):
        click.echo(f"Artifacts saved in: {result['run_root']}")


@cli.command(name="train-matrix")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def train_matrix(config_path: str) -> None:
    """Run a predefined matrix of training configs."""
    run_matrix(config_path)


@cli.command(name="train-roadmap")
@click.option(
    "--stage",
    type=click.Choice(["all", "freeze-eval", "a", "b", "c", "d", "e"]),
    default="all",
)
@click.option("--protocol-config", default="configs/retrieval/eval_protocol_oscar.yaml", type=click.Path(exists=True))
@click.option("--no-baselines", is_flag=True, default=False)
def train_roadmap(stage: str, protocol_config: str, no_baselines: bool) -> None:
    """Run the staged OSCAR retrieval experiment roadmap."""
    run_roadmap(stage=stage, protocol_config=protocol_config, run_baselines=not no_baselines)


@cli.command(name="build-final-report")
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--contrastive-summary", required=True, type=click.Path(exists=False))
@click.option("--distill-summary", required=True, type=click.Path(exists=False))
@click.option("--two-stage-summary", required=True, type=click.Path(exists=False))
@click.option("--lora-summary", required=True, type=click.Path(exists=False))
@click.option("--beir-summary", required=True, type=click.Path(exists=False))
def build_final_report_cmd(
    output_dir: str,
    contrastive_summary: str,
    distill_summary: str,
    two_stage_summary: str,
    lora_summary: str,
    beir_summary: str,
) -> None:
    """Aggregate matrix + BEIR summaries into final report."""
    build_final_report(
        output_dir=output_dir,
        contrastive_summary=contrastive_summary,
        distill_summary=distill_summary,
        two_stage_summary=two_stage_summary,
        lora_summary=lora_summary,
        beir_summary=beir_summary,
    )


@cli.command(name="aggregate-beir3")
@click.option("--summary", "summary_paths", multiple=True, required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
def aggregate_beir3_cmd(summary_paths: tuple[str, ...], output_dir: str) -> None:
    """Aggregate multiple BEIR3 summaries into comparison artifacts."""
    result = aggregate_beir3_results([Path(path) for path in summary_paths], Path(output_dir))
    click.echo(f"Aggregated {len(result['runs'])} BEIR3 run(s) into: {output_dir}")


@cli.group()
def analysis() -> None:
    """Run offline analysis and diagnostic jobs."""


@analysis.command(name="mem-latents")
@click.option("--beir-config", required=True, type=click.Path(exists=True))
@click.option("--oscar-model", required=True)
@click.option("--output-dir", default="artifacts/analysis/mem_latent_diagnostics", type=click.Path())
@click.option("--poolers", default=",".join(DEFAULT_POOLERS), show_default=True)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--max-docs-for-cosine", type=int, default=1000, show_default=True)
@click.option("--teacher-h5", multiple=True, type=click.Path())
@click.option("--teacher-max-samples", type=int, default=2000, show_default=True)
@click.option("--teacher-train-fraction", type=float, default=0.8, show_default=True)
@click.option("--torch-dtype", default="bfloat16", show_default=True)
@click.option("--no-trust-remote-code", is_flag=True, default=False)
@click.option("--seed", type=int, default=42, show_default=True)
def analysis_mem_latents(
    beir_config: str,
    oscar_model: str,
    output_dir: str,
    poolers: str,
    device: str,
    batch_size: int,
    max_docs_for_cosine: int,
    teacher_h5: tuple[str, ...],
    teacher_max_samples: int,
    teacher_train_fraction: float,
    torch_dtype: str,
    no_trust_remote_code: bool,
    seed: int,
) -> None:
    """Diagnose raw OSCAR MEM latents with pooling, probes, and BEIR retrieval."""
    summary = run_mem_latent_diagnostics(
        beir_config_path=Path(beir_config),
        oscar_model=oscar_model,
        output_dir=Path(output_dir),
        poolers=poolers.split(","),
        device=device,
        batch_size=batch_size,
        max_docs_for_cosine=max_docs_for_cosine,
        teacher_h5=list(teacher_h5),
        teacher_max_samples=teacher_max_samples,
        teacher_train_fraction=teacher_train_fraction,
        torch_dtype_name=torch_dtype,
        trust_remote_code=not no_trust_remote_code,
        seed=seed,
    )
    click.echo(f"MEM latent diagnostics saved in: {output_dir}")
    click.echo(f"Run id: {summary['run_id']}")


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


@retrieval.command(name="evaluate-beir")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def retrieval_evaluate_beir(config_path: str) -> None:
    """Evaluate retrieval checkpoints on BEIR-style datasets."""
    config = load_yaml(config_path)
    evaluate_beir(config)


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


@data.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, name="leaf-corpus")
@click.pass_context
def data_leaf_corpus(ctx: click.Context) -> None:
    _run_recipe_module("projected_token.data.recipes.prepare_leaf_corpus", tuple(ctx.args))


@data.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, name="leaf-teacher-embeddings")
@click.pass_context
def data_leaf_teacher_embeddings(ctx: click.Context) -> None:
    _run_recipe_module("projected_token.data.recipes.generate_leaf_teacher_embeddings", tuple(ctx.args))


@data.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, name="query-doc-teacher-embeddings")
@click.pass_context
def data_query_doc_teacher_embeddings(ctx: click.Context) -> None:
    _run_recipe_module("projected_token.data.recipes.generate_query_doc_teacher_embeddings", tuple(ctx.args))


@data.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True}, name="teacher-embeddings")
@click.pass_context
def data_teacher_embeddings(ctx: click.Context) -> None:
    _run_recipe_module("projected_token.data.recipes.generate_teacher_embeddings", tuple(ctx.args))
