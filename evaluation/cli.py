import click
import yaml
import re
from typing import Dict, Any, List, Optional
import json
import sys
import traceback
from datetime import datetime
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path

# Factory function to instantiate classes from config
def instantiate_class(class_config: Dict[str, Any], class_type: str = "class") -> Any:
    """Instantiate a class from configuration dictionary."""
    if 'class' not in class_config:
        raise ValueError(f"Missing 'class' key in {class_type} configuration")
    
    # Import the class
    class_path = class_config['class']
    if '.' not in class_path:
        raise ValueError(f"Invalid class path: {class_path}. Expected format: 'module.submodule.ClassName'")
    
    module_name, class_name = class_path.rsplit('.', 1)
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {class_path}: {e}")
    
    # Get arguments (if any)
    args = class_config.get('args', [])
    kwargs = class_config.get('kwargs', {})
    
    return cls(*args, **kwargs)

def log_step(message: str, level: str = "info"):
    """Log a step with pretty formatting."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if level == "info":
        click.echo(click.style(f"[{timestamp}] ℹ️  ", fg="blue") + click.style(message, fg="white"))
    elif level == "success":
        click.echo(click.style(f"[{timestamp}] ✅ ", fg="green") + click.style(message, fg="white"))
    elif level == "warning":
        click.echo(click.style(f"[{timestamp}] ⚠️  ", fg="yellow") + click.style(message, fg="white"))
    elif level == "error":
        click.echo(click.style(f"[{timestamp}] ❌ ", fg="red") + click.style(message, fg="white"))
    elif level == "start":
        click.echo(click.style(f"[{timestamp}] 🚀 ", fg="cyan") + click.style(message, fg="white", bold=True))

def log_substep(message: str):
    """Log a substep with indentation."""
    click.echo(click.style("  ↳ ", fg="blue") + message)


# Simple metrics functions (from evaluate_gpt_judge_only.py)
def simple_tokenize(text: str):
    """Simple tokenization by splitting on whitespace."""
    return text.lower().split()


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def calculate_simple_metrics(originals: List[str], paraphrases: List[str]) -> Dict[str, float]:
    """Calculate simple text similarity metrics."""
    results = []

    for orig, para in zip(originals, paraphrases):
        if not orig.strip() or not para.strip():
            continue

        orig_tokens = set(simple_tokenize(orig))
        para_tokens = set(simple_tokenize(para))

        jaccard = jaccard_similarity(orig_tokens, para_tokens)
        char_sim = SequenceMatcher(None, orig.lower(), para.lower()).ratio()
        word_overlap = len(orig_tokens & para_tokens) / max(len(orig_tokens), 1)
        orig_len = len(orig.split())
        para_len = len(para.split())
        length_ratio = para_len / max(orig_len, 1)

        results.append({
            'jaccard': jaccard,
            'char_sim': char_sim,
            'word_overlap': word_overlap,
            'length_ratio': length_ratio
        })

    if not results:
        return {}

    avg_metrics = {}
    for key in ['jaccard', 'char_sim', 'word_overlap', 'length_ratio']:
        avg_metrics[f'avg_{key}'] = float(np.mean([r[key] for r in results]))

    return avg_metrics


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Click CLI
@click.group()
def cli():
    """CLI for paraphrasing and QA tasks."""
    pass

def _process_task(config, input_path, output_path, text_col, output_col, verbose, task_type, batch_size: Optional[int] = None):
    """Common processing function for both paraphrase and QA tasks with optional batching."""
    task_name = "paraphrase" if task_type == "paraphrase" else "QA"
    log_step(f"Starting {task_name} operation", "start")
    
    try:
        # Load configuration
        log_step("Loading configuration file")
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        log_substep(f"Config loaded from {config}")
        
        # Instantiate dataset
        log_step("Instantiating dataset")
        dataset_config = config_data['dataset']
        dataset = instantiate_class(dataset_config, "dataset")
        dataset_instance = dataset.load(input_path)
        log_substep(f"Dataset loaded from {input_path}")
        
        # Instantiate model
        log_step("Instantiating model")
        model_config = config_data['model']
        model = instantiate_class(model_config, "model")
        log_substep(f"Model initialized: {model_config.get('class', 'Unknown')}")
        
        # Check if model supports batch inference
        supports_batch = hasattr(model, 'generate_batch') and callable(getattr(model, 'generate_batch'))
        if batch_size and not supports_batch:
            log_step("Model doesn't support batch inference, falling back to single item processing", "warning")
            batch_size = None
        
        # Get model generation parameters from config
        model_args = config_data.get('generation', {})
        log_substep(f"Generation parameters: {model_args}")
        
        experiment_config = config_data.get('experiment', {})
        log_substep(f"Experiment parameters: {experiment_config}")
        
        # Get appropriate prompt template based on task type
        if task_type == "paraphrase":
            prompt_template = (
                experiment_config.get('prompt_template')
                or getattr(model, 'PARAPHRASE_DEFAULT_PROMPT_TEMPLATE', None)
                or getattr(model, 'default_prompt_template', None)
            )
        else:  # QA task
            prompt_template = (
                experiment_config.get('prompt_template')
                or getattr(model, 'QA_DEFAULT_PROMPT_TEMPLATE', None)
                or getattr(model, 'default_prompt_template', None)
            )
        
        if not prompt_template:
            raise ValueError(f"No prompt template found for {task_type} task")
            
        log_substep(f"Prompt template: `{prompt_template}`")
        
        # Process data
        log_step("Processing data")
        output_data = []
        processed_count = 0
        error_count = 0
        
        if batch_size and supports_batch:
            # Batch processing mode
            log_substep(f"Using batch processing with size {batch_size}")
            batch_items, batch_docs, batch_questions = [], [], []
            
            def flush_batch():
                nonlocal batch_items, batch_docs, batch_questions, output_data, processed_count, error_count
                if not batch_docs:
                    return
                
                items_expanded = []
                docs_expanded = []
                questions_expanded = []
                
                for i, (item, doc) in enumerate(zip(batch_items, batch_docs)):
                    sample_count = int(experiment_config.get('sample_count', 1))
                    for _ in range(sample_count):
                        items_expanded.append(item)
                        docs_expanded.append(doc)
                        if task_type == "qa":
                            questions_expanded.append(batch_questions[i])
                
                try:
                    if task_type == "paraphrase":
                        outputs = model.generate_batch(docs_expanded, prompt_template, model_args)
                    else:  # QA task
                        # For QA, pass documents and questions separately for models that support it (e.g., XRAG)
                        # Falls back to formatted prompts for models that don't support questions param
                        import inspect
                        sig = inspect.signature(model.generate_batch)
                        if 'questions' in sig.parameters:
                            outputs = model.generate_batch(docs_expanded, prompt_template, model_args, questions=questions_expanded)
                        else:
                            # Fallback: format prompts manually
                            formatted_prompts = []
                            for doc, question in zip(docs_expanded, questions_expanded):
                                formatted_prompts.append(prompt_template.format(document=doc, question=question))
                            outputs = model.generate_batch(formatted_prompts, "", model_args)
                    
                    if len(outputs) != len(items_expanded):
                        raise RuntimeError(
                            f"Batch output size mismatch: got {len(outputs)} outputs for "
                            f"{len(items_expanded)} requested generations."
                        )
                    
                    for item, out in zip(items_expanded, outputs):
                        if task_type == "paraphrase":
                            output_item = item | {output_col: out}
                        else:  # QA task
                            output_item = item | {'answer': out}
                        output_data.append(output_item)
                        processed_count += 1
                        
                        if verbose and processed_count % 10 == 0:
                            log_substep(f"Sample output: {str(out)[:100]}...")
                            
                except Exception as e:
                    # If a whole batch fails, fall back to per-item inference
                    if verbose:
                        log_step(f"Batch failed ({len(batch_docs)} items): {e}", "warning")
                    
                    for idx_in_batch, (item, doc) in enumerate(zip(batch_items, batch_docs)):
                        try:
                            sample_count = int(experiment_config.get('sample_count', 1))
                            for _ in range(sample_count):
                                if task_type == "paraphrase":
                                    formatted_prompt = prompt_template.format(document=doc)
                                    out = model(doc, formatted_prompt, model_args)
                                else:  # QA task
                                    question = batch_questions[idx_in_batch]
                                    formatted_prompt = prompt_template.format(document=doc, question=question)
                                    out = model(formatted_prompt, model_args)
                                
                                if task_type == "paraphrase":
                                    output_item = item | {output_col: out}
                                else:  # QA task
                                    output_item = item | {'answer': out}
                                output_data.append(output_item)
                                processed_count += 1
                                
                        except Exception as e2:
                            error_count += 1
                            if verbose:
                                log_step(f"Error processing item in failed batch (idx={idx_in_batch}): {e2}", "warning")
                finally:
                    batch_items, batch_docs, batch_questions = [], [], []
            
            for i, item in tqdm(enumerate(dataset_instance)):
                if verbose and i % 100 == 0:
                    log_substep(f"Processed {i} items...")
                
                document = item.get(text_col, '')
                if not document or not isinstance(document, str):
                    if verbose:
                        log_step(f"Skipping item {i}: missing text column '{text_col}' or not a string", "warning")
                    error_count += 1
                    continue
                
                if task_type == "qa":
                    question = item.get('question', '')
                    if not question:
                        if verbose:
                            log_step(f"Skipping item {i}: missing 'question' column for QA task", "warning")
                        error_count += 1
                        continue
                    batch_questions.append(question)
                
                batch_items.append(item)
                batch_docs.append(document)
                
                if len(batch_docs) >= batch_size:
                    flush_batch()
            
            flush_batch()  # Process any remaining items
            
        else:
            # Single item processing mode
            log_substep("Using single item processing")
            for i, item in tqdm(enumerate(dataset_instance)):
                if verbose and i % 100 == 0:
                    log_substep(f"Processed {i} items...")
                
                document = item.get(text_col, '')
                if not document or not isinstance(document, str):
                    if verbose:
                        log_step(f"Skipping item {i}: missing text column '{text_col}' or not a string", "warning")
                    error_count += 1
                    continue
                
                try:
                    sample_count = int(experiment_config.get('sample_count', 1))
                    for _ in range(sample_count):
                        if task_type == "paraphrase":
                            formatted_prompt = prompt_template.format(document=document)
                            result = model(document, formatted_prompt, model_args)
                            output_item = item | {output_col: result}
                        else:  # QA task
                            question = item.get('question', '')
                            if not question:
                                log_step(f"Skipping item {i}: missing 'question' column for QA task", "warning")
                                error_count += 1
                                continue
                            formatted_prompt = prompt_template.format(document=document, question=question)
                            result = model(formatted_prompt, model_args)
                            output_item = item | {'answer': result}
                        
                        output_data.append(output_item)
                        processed_count += 1
                        
                        if verbose and processed_count % 10 == 0:
                            log_substep(f"Sample output: {result[:100]}...")
                            
                except Exception as e:
                    if verbose:
                        log_step(f"Error processing item {i}: {str(e)}", "warning")
                        log_substep(f"Document: {document[:100]}...")
                    error_count += 1
                    continue
        
        # Save results
        log_step("Saving results")
        with open(output_path, 'w') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        log_step(f"{task_name.capitalize()} completed successfully!", "success")
        log_substep(f"Processed: {processed_count} items")
        log_substep(f"Errors: {error_count} items")
        log_substep(f"Output: {output_path}")
        
    except Exception as e:
        log_step(f"Fatal error: {str(e)}", "error")
        if verbose:
            click.echo(click.style("Stack trace:", fg="red"))
            click.echo(traceback.format_exc())
        sys.exit(1)

@cli.command()
@click.option('--config', required=True, help='Path to YAML configuration file')
@click.option('--input-path', required=True, help='Input dataset path')
@click.option('--output-path', required=True, help='Output file path')
@click.option('--text-col', default='text', help='Column name for input text')
@click.option('--output-col', default='rephrased_text', help='Column name for rephrased text')
@click.option('--batch-size', default=None, type=int, help='Batch size for model inference (None for single item)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def paraphrase(config, input_path, output_path, text_col, output_col, batch_size, verbose):
    """Rephrase text using a model and save results."""
    _process_task(config, input_path, output_path, text_col, output_col, verbose, "paraphrase", batch_size)

@cli.command()
@click.option('--config', required=True, help='Path to YAML configuration file')
@click.option('--input-path', required=True, help='Input dataset path with questions')
@click.option('--output-path', required=True, help='Output file path with answers')
@click.option('--text-col', default='text', help='Column name for context text')
@click.option('--batch-size', default=None, type=int, help='Batch size for model inference (None for single item)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def qa(config, input_path, output_path, text_col, batch_size, verbose):
    """Answer questions using a model and save results."""
    _process_task(config, input_path, output_path, text_col, 'answer', verbose, "qa", batch_size)

@cli.command()
@click.option('--config', required=True, help='Path to YAML configuration file')
@click.option('--input-path', required=True, help='Input dataset path with rephrased text')
@click.option('--output-path', required=True, help='Output file path with metrics')
@click.option('--original-col', default='text', help='Column name for original text')
@click.option('--rephrased-col', default='rephrased_text', help='Column name for rephrased text')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def evaluate_paraphrase(config, input_path, output_path, original_col, rephrased_col, verbose):
    """Evaluate rephrased text using metrics."""
    log_step("Starting evaluation operation", "start")
    
    try:
        # Load configuration
        log_step("Loading configuration file")
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        log_substep(f"Config loaded from {config}")
        
        # Instantiate dataset
        log_step("Instantiating dataset")
        dataset_config = config_data['dataset']
        dataset = instantiate_class(dataset_config, "dataset")
        dataset_instance = dataset.load(input_path)
        log_substep(f"Dataset loaded from {input_path}")
        
        # Instantiate metrics
        log_step("Instantiating metrics")
        metrics_config = config_data.get('metrics', [])
        metrics = []
        for i, metric_config in enumerate(metrics_config):
            try:
                metric = instantiate_class(metric_config, f"metric_{i}")
                metrics.append(metric)
                log_substep(f"Metric {i+1}: {metric_config.get('class', 'Unknown')}")
            except Exception as e:
                log_step(f"Failed to instantiate metric {i}: {e}", "warning")
        
        if not metrics:
            raise ValueError("No metrics were successfully instantiated")
        
        # Process data
        log_step("Evaluating data")
        output_data = []
        evaluated_count = 0
        error_count = 0
        
        for i, item in enumerate(dataset_instance):
            if verbose and i % 100 == 0:
                log_substep(f"Evaluated {i} items...")
            
            original_text = item.get(original_col, '')
            rephrased_text = item.get(rephrased_col, '')
            
            if not original_text or not rephrased_text:
                if verbose:
                    log_step(f"Skipping item {i}: missing text columns", "warning")
                error_count += 1
                continue
            
            metrics_results = {}
            metric_errors = 0
            
            for j, metric in enumerate(metrics):
                try:
                    result = metric(original_text, rephrased_text)
                    metrics_results.update(result)
                    if verbose and j == 0:  # Log first metric result as sample
                        log_substep(f"Sample metrics: {list(result.keys())}")
                except Exception as e:
                    if verbose:
                        log_step(f"Metric {j+1} failed on item {i}: {e}", "warning")
                    metric_errors += 1
            
            if metric_errors < len(metrics):  # Only add if at least one metric succeeded
                item['metrics'] = metrics_results
                output_data.append(item)
                evaluated_count += 1
            else:
                error_count += 1
        
        # Save results
        log_step("Saving evaluation results")
        with open(output_path, 'w') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        log_step(f"Evaluation completed successfully!", "success")
        log_substep(f"Evaluated: {evaluated_count} items")
        log_substep(f"Errors: {error_count} items")
        log_substep(f"Output: {output_path}")
        
    except Exception as e:
        log_step(f"Fatal error: {str(e)}", "error")
        if verbose:
            click.echo(click.style("Stack trace:", fg="red"))
            click.echo(traceback.format_exc())
        sys.exit(1)

# =============================================================================
# GPT Judge Evaluation Command
# =============================================================================

@cli.command()
@click.option('--input-path', required=True, help='Path to JSONL file with paraphrases')
@click.option('--output', '-o', required=True, help='Output path for metrics JSON')
@click.option('--base-url', default='http://localhost:8000/v1', help='vLLM API base URL')
@click.option('--api-key', default='dummy', help='API key (can be dummy for local)')
@click.option('--model', default='Qwen/Qwen3.5-27B', help='Model name for judge')
@click.option('--batch-size', type=int, default=10, help='Batch size for GPT Judge')
@click.option('--original-col', default='s_wiki_content', help='Column name for original text')
@click.option('--rephrased-col', default='rephrased_text', help='Column name for paraphrased text')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def eval_gpt(input_path, output, base_url, api_key, model, batch_size, original_col, rephrased_col, verbose):
    """Evaluate paraphrases using GPT Judge and simple metrics."""
    from openai import OpenAI
    import instructor
    from pydantic import BaseModel, Field, confloat
    from enum import Enum

    class Label(str, Enum):
        supported = "supported"
        partially_supported = "partially_supported"
        contradicted = "contradicted"
        unknown = "unknown"

    class FactualityVerdict(BaseModel):
        supported_claims: List[str] = Field(default_factory=list)
        contradicted_claims: List[str] = Field(default_factory=list)
        not_in_reference: List[str] = Field(default_factory=list)
        rationale: str = Field(description="Brief explanation")
        score: confloat(ge=0.0, le=1.0) = Field(description="0..1 factual consistency")
        label: Label

    class VerdictWithId(FactualityVerdict):
        id: int = Field(description="Must match the input case id")

    class BatchVerdicts(BaseModel):
        verdicts: List[VerdictWithId]

    SYSTEM_PROMPT = """You are a strict factual consistency judge.

You will be given a JSON array called "cases".
Each case has:
- id (int)
- candidate (Text A)
- reference (Text B)

Task:
- For each case, compare candidate against reference.
- Judge whether the factual statements in candidate are supported by reference.

Rubric:
- supported: All factual claims in A are supported by B.
- partially_supported: Most claims supported, but A has minor unsupported/ambiguous parts.
- contradicted: Any clear factual contradiction between A and B.
- unknown: B lacks enough info to assess most claims in A.

Guidelines:
- Treat reference as the only ground truth.
- If A adds details not present in B, list them under not_in_reference.
- If A conflicts with B, list conflicts under contradicted_claims.
- Keep claims short and atomic when listing.
- Return one verdict per input case.

Return only the structured output with:
{ "verdicts": [ ... ] }"""

    log_step("Starting GPT Judge evaluation", "start")

    try:
        # Load data
        log_step("Loading input data")
        data = load_jsonl(input_path)
        log_substep(f"Loaded {len(data)} items from {input_path}")

        # Extract originals and paraphrases
        originals = [d.get(original_col, '') for d in data]
        paraphrases = [d.get(rephrased_col, '') for d in data]

        # Filter valid pairs
        valid_indices = [i for i in range(len(data)) if originals[i].strip() and paraphrases[i].strip()]
        originals = [originals[i] for i in valid_indices]
        paraphrases = [paraphrases[i] for i in valid_indices]

        log_substep(f"{len(originals)} valid paraphrase pairs")

        # Calculate simple metrics
        log_step("Calculating simple metrics")
        simple_metrics = calculate_simple_metrics(originals, paraphrases)
        log_substep(f"avg_jaccard: {simple_metrics.get('avg_jaccard', 0):.4f}")
        log_substep(f"avg_char_sim: {simple_metrics.get('avg_char_sim', 0):.4f}")
        log_substep(f"avg_word_overlap: {simple_metrics.get('avg_word_overlap', 0):.4f}")
        log_substep(f"avg_length_ratio: {simple_metrics.get('avg_length_ratio', 0):.4f}")

        # Initialize GPT Judge
        log_step(f"Initializing GPT Judge (model: {model})")
        raw_client = OpenAI(base_url=base_url, api_key=api_key)
        client = instructor.patch(raw_client, mode=instructor.Mode.JSON)

        # Run GPT Judge
        log_step(f"Running GPT Judge (batch_size={batch_size})")
        all_scores = []
        all_labels = []

        for i in tqdm(range(0, len(originals), batch_size), desc="GPT Judge batches"):
            batch_end = min(i + batch_size, len(originals))
            batch_refs = originals[i:batch_end]
            batch_cands = paraphrases[i:batch_end]

            cases = [{"id": j, "candidate": cand, "reference": ref}
                     for j, (cand, ref) in enumerate(zip(batch_cands, batch_refs))]

            try:
                response: BatchVerdicts = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    response_model=BatchVerdicts,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps({"cases": cases}, ensure_ascii=False)},
                    ],
                )

                for verdict in response.verdicts:
                    score = float(verdict.score)
                    if score >= 0:
                        all_scores.append(score)
                        all_labels.append(verdict.label.value)

            except Exception as e:
                log_step(f"Batch {i // batch_size} failed: {e}", "warning")
                # Add error scores
                for _ in range(len(batch_cands)):
                    all_scores.append(-1.0)

        # Calculate GPT metrics
        valid_scores = [s for s in all_scores if s >= 0]
        if valid_scores:
            log_step("GPT Judge Results")
            log_substep(f"avg_gpt_judge_score: {np.mean(valid_scores):.4f}")
            log_substep(f"std_gpt_judge_score: {np.std(valid_scores):.4f}")
            log_substep(f"min_gpt_judge_score: {np.min(valid_scores):.4f}")
            log_substep(f"max_gpt_judge_score: {np.max(valid_scores):.4f}")

            label_counts = {}
            for label in all_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            log_substep("Label distribution:")
            for label, count in sorted(label_counts.items()):
                log_substep(f"  {label}: {count} ({100*count/len(all_labels):.1f}%)")

            gpt_metrics = {
                'avg_gpt_judge_score': float(np.mean(valid_scores)),
                'std_gpt_judge_score': float(np.std(valid_scores)),
                'min_gpt_judge_score': float(np.min(valid_scores)),
                'max_gpt_judge_score': float(np.max(valid_scores)),
                'label_distribution': label_counts
            }
        else:
            log_step("ERROR: No valid GPT scores obtained", "error")
            gpt_metrics = {'error': 'No valid scores'}

        # Combine all metrics
        all_metrics = {**simple_metrics, **gpt_metrics}

        # Save results
        log_step("Saving metrics")
        with open(output, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        log_substep(f"Saved to {output}")

        log_step("GPT Judge evaluation completed!", "success")

    except Exception as e:
        log_step(f"Fatal error: {str(e)}", "error")
        if verbose:
            click.echo(click.style("Stack trace:", fg="red"))
            click.echo(traceback.format_exc())
        sys.exit(1)


# =============================================================================
# Run Experiment Command (paraphrase + eval-gpt)
# =============================================================================

@cli.command()
@click.option('--config', required=True, help='Path to YAML configuration file')
@click.option('--input-path', required=True, help='Input dataset path')
@click.option('--output-dir', required=True, help='Output directory for results')
@click.option('--text-col', default='s_wiki_content', help='Column name for input text')
@click.option('--batch-size', type=int, default=5, help='Batch size for model inference')
@click.option('--gpt-base-url', default='http://localhost:8000/v1', help='vLLM API base URL for GPT Judge')
@click.option('--gpt-model', default='Qwen/Qwen3.5-27B', help='Model name for GPT Judge')
@click.option('--gpt-batch-size', type=int, default=10, help='Batch size for GPT Judge')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def run_experiment(config, input_path, output_dir, text_col, batch_size, gpt_base_url, gpt_model, gpt_batch_size, verbose):
    """Run a full experiment: paraphrase + GPT Judge evaluation."""
    import subprocess
    from datetime import datetime

    log_step("Starting full experiment", "start")

    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        log_substep(f"Output directory: {output_path}")

        # Get model name from config for output filenames
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        model_class = config_data.get('model', {}).get('class', 'unknown')
        model_name = model_class.split('.')[-1].replace('Model', '').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        paraphrase_output = output_path / f"{model_name}_paraphrase_{timestamp}.jsonl"
        metrics_output = output_path / f"{model_name}_metrics_{timestamp}.json"

        log_substep(f"Model: {model_name}")
        log_substep(f"Paraphrase output: {paraphrase_output}")
        log_substep(f"Metrics output: {metrics_output}")

        # Step 1: Run paraphrase
        log_step("Step 1: Running paraphrase")
        paraphrase_cmd = [
            sys.executable, '-m', 'evaluation.cli', 'paraphrase',
            '--config', str(config),
            '--input-path', str(input_path),
            '--output-path', str(paraphrase_output),
            '--text-col', text_col,
            '--batch-size', str(batch_size),
        ]
        if verbose:
            paraphrase_cmd.append('--verbose')

        result = subprocess.run(paraphrase_cmd)
        if result.returncode != 0:
            log_step("Paraphrase failed!", "error")
            sys.exit(1)
        log_step("Paraphrase completed!", "success")

        # Step 2: Run GPT Judge evaluation
        log_step("Step 2: Running GPT Judge evaluation")
        eval_cmd = [
            sys.executable, '-m', 'evaluation.cli', 'eval-gpt',
            '--input-path', str(paraphrase_output),
            '--output', str(metrics_output),
            '--base-url', gpt_base_url,
            '--model', gpt_model,
            '--batch-size', str(gpt_batch_size),
            '--original-col', text_col,
            '--rephrased-col', 'rephrased_text',
        ]
        if verbose:
            eval_cmd.append('--verbose')

        result = subprocess.run(eval_cmd)
        if result.returncode != 0:
            log_step("GPT Judge evaluation failed!", "error")
            sys.exit(1)
        log_step("GPT Judge evaluation completed!", "success")

        log_step("Full experiment completed!", "success")
        log_substep(f"Paraphrase: {paraphrase_output}")
        log_substep(f"Metrics: {metrics_output}")

    except Exception as e:
        log_step(f"Fatal error: {str(e)}", "error")
        if verbose:
            click.echo(click.style("Stack trace:", fg="red"))
            click.echo(traceback.format_exc())
        sys.exit(1)


# =============================================================================
# Run QA Experiment Command (QA generation + QA evaluation)
# =============================================================================

@cli.command()
@click.option('--config', required=True, help='Path to YAML configuration file')
@click.option('--input-path', required=True, help='Input dataset path (PopQA parquet)')
@click.option('--output-dir', required=True, help='Output directory for results')
@click.option('--text-col', default='s_wiki_content', help='Column name for context text')
@click.option('--question-col', default='question', help='Column name for questions')
@click.option('--answer-col', default='possible_answers', help='Column name for ground truth answers')
@click.option('--batch-size', type=int, default=5, help='Batch size for model inference')
@click.option('--gpt-base-url', default='http://localhost:8000/v1', help='vLLM API base URL')
@click.option('--gpt-api-key', default='dummy', help='API key for QA evaluation (can be dummy for local)')
@click.option('--gpt-model', default='Qwen/Qwen3.5-27B', help='Model for QA evaluation')
@click.option('--gpt-batch-size', type=int, default=10, help='Batch size for QA evaluation')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def run_qa_experiment(config, input_path, output_dir, text_col, question_col,
                      answer_col, batch_size, gpt_base_url, gpt_api_key,
                      gpt_model, gpt_batch_size, verbose):
    """Run a full QA experiment: generate answers + LLM-based evaluation."""
    import subprocess
    from datetime import datetime

    log_step("Starting QA experiment", "start")

    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        log_substep(f"Output directory: {output_path}")

        # Get model name from config for output filenames
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        model_class = config_data.get('model', {}).get('class', 'unknown')
        model_name = model_class.split('.')[-1].replace('Model', '').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        qa_output = output_path / f"{model_name}_qa_answers_{timestamp}.jsonl"
        metrics_output = output_path / f"{model_name}_qa_metrics_{timestamp}.json"

        log_substep(f"Model: {model_name}")
        log_substep(f"QA output: {qa_output}")
        log_substep(f"Metrics output: {metrics_output}")

        # Step 1: Run QA generation
        log_step("Step 1: Running QA generation")
        qa_cmd = [
            sys.executable, '-m', 'evaluation.cli', 'qa',
            '--config', str(config),
            '--input-path', str(input_path),
            '--output-path', str(qa_output),
            '--text-col', text_col,
            '--batch-size', str(batch_size),
        ]
        if verbose:
            qa_cmd.append('--verbose')

        result = subprocess.run(qa_cmd)
        if result.returncode != 0:
            log_step("QA generation failed!", "error")
            sys.exit(1)
        log_step("QA generation completed!", "success")

        # Step 2: Run QA evaluation
        log_step("Step 2: Running QA evaluation")
        eval_qa_cmd = [
            sys.executable, '-m', 'evaluation.cli', 'eval-qa',
            '--input-path', str(qa_output),
            '--output', str(metrics_output),
            '--base-url', gpt_base_url,
            '--api-key', gpt_api_key,
            '--model', gpt_model,
            '--batch-size', str(gpt_batch_size),
            '--question-col', question_col,
            '--answer-col', answer_col,
        ]
        if verbose:
            eval_qa_cmd.append('--verbose')

        result = subprocess.run(eval_qa_cmd)
        if result.returncode != 0:
            log_step("QA evaluation failed!", "error")
            sys.exit(1)
        log_step("QA evaluation completed!", "success")

        log_step("QA experiment completed!", "success")
        log_substep(f"QA answers: {qa_output}")
        log_substep(f"Metrics: {metrics_output}")

    except Exception as e:
        log_step(f"Fatal error: {str(e)}", "error")
        if verbose:
            click.echo(click.style("Stack trace:", fg="red"))
            click.echo(traceback.format_exc())
        sys.exit(1)


# =============================================================================
# QA Evaluation Command (LLM-based QA scoring)
# =============================================================================

@cli.command()
@click.option('--input-path', required=True, help='Path to JSONL file with QA answers')
@click.option('--output', '-o', required=True, help='Output path for metrics JSON')
@click.option('--base-url', default='http://localhost:8000/v1', help='vLLM API base URL')
@click.option('--api-key', default='dummy', help='API key (can be dummy for local)')
@click.option('--model', default='Qwen/Qwen3.5-27B', help='Model name for QA judge')
@click.option('--batch-size', type=int, default=10, help='Batch size for QA evaluation')
@click.option('--question-col', default='question', help='Column name for questions')
@click.option('--answer-col', default='possible_answers', help='Column name for ground truth answers')
@click.option('--predicted-col', default='answer', help='Column name for predicted answers')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def eval_qa(input_path, output, base_url, api_key, model, batch_size,
            question_col, answer_col, predicted_col, verbose):
    """Evaluate QA answers using LLM-based scoring and in_accuracy metric."""
    from openai import OpenAI
    import instructor
    from pydantic import BaseModel, Field, confloat
    from enum import Enum

    class QALabel(str, Enum):
        correct = "correct"
        partially_correct = "partially_correct"
        wrong = "wrong"

    class QAVerdict(BaseModel):
        label: QALabel = Field(description="One of: correct, partially_correct, wrong")
        score: confloat(ge=0.0, le=1.0) = Field(description="1.0 for correct, 0.5 for partially correct, 0.0 for wrong")
        rationale: str = Field(description="Brief explanation")

    class VerdictWithId(QAVerdict):
        id: int = Field(description="Must match the input case id")

    class BatchQAVerdicts(BaseModel):
        verdicts: List[VerdictWithId]

    QA_SYSTEM_PROMPT = """You are an evaluation tool. Answer with one of:
1: Correct,
0.5: Partially correct,
0: Wrong.

You will be given a JSON array called "cases".
Each case has:
- id (int)
- question (str)
- golden_answer (str) - the ground truth answer
- ai_answer (str) - the AI-generated answer to evaluate

Task:
- For each case, judge whether the AI-generated answer is correct according to the question and golden answer.

Rubric:
- correct (1): The AI answer is factually correct and matches the golden answer.
- partially_correct (0.5): The AI answer is partially correct or contains some correct information but is incomplete.
- wrong (0): The AI answer is factually incorrect or does not answer the question properly.

Guidelines:
- Treat the golden_answer as the ground truth.
- Consider semantic equivalence, not just exact string matching.
- Return one verdict per input case.

Return only the structured output with:
{ "verdicts": [ ... ] }"""

    # Text normalization for in_accuracy
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def compute_in_accuracy(predicted_answer: str, reference_answers: List[str]) -> bool:
        predicted_normalized = normalize_text(predicted_answer)
        for ref in reference_answers:
            ref_normalized = normalize_text(ref)
            if ref_normalized in predicted_normalized:
                return True
        return False

    log_step("Starting QA evaluation", "start")

    try:
        # Load data
        log_step("Loading input data")
        data = load_jsonl(input_path)
        log_substep(f"Loaded {len(data)} items from {input_path}")

        # Extract questions, reference answers, and predicted answers
        questions = []
        reference_answers_list = []
        predicted_answers = []

        for item in data:
            question = item.get(question_col, '')
            predicted = item.get(predicted_col, '')

            # Handle answer_col - could be a list or string
            ref_answers = item.get(answer_col, '')
            if isinstance(ref_answers, str):
                ref_answers = [ref_answers]
            elif ref_answers is None:
                ref_answers = []

            if question.strip() and predicted.strip():
                questions.append(question)
                reference_answers_list.append(ref_answers)
                predicted_answers.append(predicted)

        log_substep(f"{len(questions)} valid QA pairs")

        if not questions:
            log_step("No valid QA pairs found!", "error")
            sys.exit(1)

        # Initialize QA Judge
        log_step(f"Initializing QA Judge (model: {model})")
        raw_client = OpenAI(base_url=base_url, api_key=api_key)
        client = instructor.patch(raw_client, mode=instructor.Mode.JSON)

        # Run QA evaluation
        log_step(f"Running QA evaluation (batch_size={batch_size})")
        all_scores = []
        all_labels = []
        all_in_accuracy = []

        for i in tqdm(range(0, len(questions), batch_size), desc="QA evaluation batches"):
            batch_end = min(i + batch_size, len(questions))
            batch_questions = questions[i:batch_end]
            batch_ref_answers = reference_answers_list[i:batch_end]
            batch_pred_answers = predicted_answers[i:batch_end]

            # Compute in_accuracy for batch
            for ref_answers, pred_answer in zip(batch_ref_answers, batch_pred_answers):
                in_acc = compute_in_accuracy(pred_answer, ref_answers)
                all_in_accuracy.append(in_acc)

            # Create cases for LLM evaluation
            cases = []
            for j, (question, ref_answers, pred_answer) in enumerate(zip(batch_questions, batch_ref_answers, batch_pred_answers)):
                golden_answer = " OR ".join(ref_answers) if ref_answers else ""
                cases.append({
                    "id": j,
                    "question": question,
                    "golden_answer": golden_answer,
                    "ai_answer": pred_answer
                })

            try:
                response: BatchQAVerdicts = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    response_model=BatchQAVerdicts,
                    messages=[
                        {"role": "system", "content": QA_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps({"cases": cases}, ensure_ascii=False)},
                    ],
                )

                for verdict in response.verdicts:
                    score = float(verdict.score)
                    if score >= 0:
                        all_scores.append(score)
                        all_labels.append(verdict.label.value)

            except Exception as e:
                log_step(f"Batch {i // batch_size} failed: {e}", "warning")
                # Add error scores
                for _ in range(len(batch_pred_answers)):
                    all_scores.append(-1.0)

        # Calculate QA metrics
        valid_scores = [s for s in all_scores if s >= 0]
        in_accuracy_count = sum(all_in_accuracy)

        if valid_scores:
            log_step("QA Evaluation Results")
            log_substep(f"avg_qa_score: {np.mean(valid_scores):.4f}")
            log_substep(f"std_qa_score: {np.std(valid_scores):.4f}")
            log_substep(f"min_qa_score: {np.min(valid_scores):.4f}")
            log_substep(f"max_qa_score: {np.max(valid_scores):.4f}")
            log_substep(f"in_accuracy: {in_accuracy_count / len(all_in_accuracy):.4f}")

            label_counts = {}
            for label in all_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            log_substep("Score distribution:")
            for score, count in sorted(label_counts.items()):
                log_substep(f"  {score}: {count} ({100*count/len(all_labels):.1f}%)")

            qa_metrics = {
                'avg_qa_score': float(np.mean(valid_scores)),
                'std_qa_score': float(np.std(valid_scores)),
                'min_qa_score': float(np.min(valid_scores)),
                'max_qa_score': float(np.max(valid_scores)),
                'score_distribution': label_counts,
                'in_accuracy': float(in_accuracy_count / len(all_in_accuracy)),
                'total_samples': len(valid_scores)
            }
        else:
            log_step("ERROR: No valid QA scores obtained", "error")
            qa_metrics = {'error': 'No valid scores'}

        # Save results
        log_step("Saving metrics")
        with open(output, 'w') as f:
            json.dump(qa_metrics, f, indent=2)
        log_substep(f"Saved to {output}")

        log_step("QA evaluation completed!", "success")

    except Exception as e:
        log_step(f"Fatal error: {str(e)}", "error")
        if verbose:
            click.echo(click.style("Stack trace:", fg="red"))
            click.echo(traceback.format_exc())
        sys.exit(1)


# =============================================================================
# Update run-all command to support QA task type
# =============================================================================

@cli.command()
@click.option('--configs-dir', default='configs', help='Directory containing config files')
@click.option('--input-path', required=True, help='Input dataset path')
@click.option('--output-dir', required=True, help='Output directory for results')
@click.option('--text-col', default='s_wiki_content', help='Column name for input text')
@click.option('--question-col', default='question', help='Column name for questions')
@click.option('--answer-col', default='possible_answers', help='Column name for ground truth answers')
@click.option('--batch-size', type=int, default=5, help='Batch size for model inference')
@click.option('--gpt-base-url', default='http://localhost:8000/v1', help='vLLM API base URL')
@click.option('--gpt-api-key', default='dummy', help='API key (can be dummy for local)')
@click.option('--gpt-model', default='Qwen/Qwen3.5-27B', help='Model name for evaluation')
@click.option('--gpt-batch-size', type=int, default=10, help='Batch size for evaluation')
@click.option('--task-type', default='paraphrase',
              type=click.Choice(['paraphrase', 'qa']),
              help='Type of experiment to run')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config-filter', default=None, help='Optional regex filter for config files')
def run_all(configs_dir, input_path, output_dir, text_col, question_col, answer_col,
            batch_size, gpt_base_url, gpt_api_key, gpt_model, gpt_batch_size,
            task_type, verbose, config_filter):
    """Run all experiments for all configs in the configs directory.

    Supports both paraphrase (factual consistency) and QA (answer correctness) tasks.
    """
    import subprocess
    import re
    from datetime import datetime

    log_step(f"Starting ALL {task_type.upper()} experiments", "start")

    try:
        # Find all config files
        configs_path = Path(configs_dir)
        if not configs_path.exists():
            log_step(f"Configs directory not found: {configs_path}", "error")
            sys.exit(1)

        config_files = sorted(configs_path.glob("*.yaml"))
        if config_filter:
            pattern = re.compile(config_filter)
            config_files = [f for f in config_files if pattern.search(f.name)]

        if not config_files:
            log_step("No config files found!", "error")
            sys.exit(1)

        log_substep(f"Found {len(config_files)} config files")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Track results
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for config_file in config_files:
            log_step(f"Processing: {config_file.name}", "start")

            try:
                # Get model name from config
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                model_class = config_data.get('model', {}).get('class', 'unknown')
                model_name = model_class.split('.')[-1].replace('Model', '').lower()

                model_output_dir = output_path / model_name
                model_output_dir.mkdir(parents=True, exist_ok=True)

                if task_type == 'paraphrase':
                    # Paraphrase experiment
                    output_file = model_output_dir / f"{model_name}_paraphrase_{timestamp}.jsonl"
                    metrics_file = model_output_dir / f"{model_name}_metrics_{timestamp}.json"

                    # Run paraphrase
                    cmd = [
                        sys.executable, '-m', 'evaluation.cli', 'paraphrase',
                        '--config', str(config_file),
                        '--input-path', str(input_path),
                        '--output-path', str(output_file),
                        '--text-col', text_col,
                        '--batch-size', str(batch_size),
                    ]
                    if verbose:
                        cmd.append('--verbose')

                    result = subprocess.run(cmd)
                    if result.returncode != 0:
                        log_step(f"Paraphrase failed for {model_name}", "error")
                        results[model_name] = {"paraphrase": False, "evaluation": False, "error": "paraphrase_failed"}
                        continue

                    # Run GPT Judge
                    eval_cmd = [
                        sys.executable, '-m', 'evaluation.cli', 'eval-gpt',
                        '--input-path', str(output_file),
                        '--output', str(metrics_file),
                        '--base-url', gpt_base_url,
                        '--api-key', gpt_api_key,
                        '--model', gpt_model,
                        '--batch-size', str(gpt_batch_size),
                        '--original-col', text_col,
                        '--rephrased-col', 'rephrased_text',
                    ]
                    if verbose:
                        eval_cmd.append('--verbose')

                    result = subprocess.run(eval_cmd)
                    if result.returncode != 0:
                        log_step(f"GPT Judge failed for {model_name}", "error")
                        results[model_name] = {"paraphrase": True, "evaluation": False, "error": "evaluation_failed"}
                        continue

                    results[model_name] = {"paraphrase": True, "evaluation": True, "metrics_file": str(metrics_file)}

                else:  # task_type == 'qa'
                    # QA experiment
                    output_file = model_output_dir / f"{model_name}_qa_answers_{timestamp}.jsonl"
                    metrics_file = model_output_dir / f"{model_name}_qa_metrics_{timestamp}.json"

                    # Run QA generation
                    cmd = [
                        sys.executable, '-m', 'evaluation.cli', 'qa',
                        '--config', str(config_file),
                        '--input-path', str(input_path),
                        '--output-path', str(output_file),
                        '--text-col', text_col,
                        '--batch-size', str(batch_size),
                    ]
                    if verbose:
                        cmd.append('--verbose')

                    result = subprocess.run(cmd)
                    if result.returncode != 0:
                        log_step(f"QA generation failed for {model_name}", "error")
                        results[model_name] = {"qa": False, "evaluation": False, "error": "qa_failed"}
                        continue

                    # Run QA evaluation
                    eval_cmd = [
                        sys.executable, '-m', 'evaluation.cli', 'eval-qa',
                        '--input-path', str(output_file),
                        '--output', str(metrics_file),
                        '--base-url', gpt_base_url,
                        '--api-key', gpt_api_key,
                        '--model', gpt_model,
                        '--batch-size', str(gpt_batch_size),
                        '--question-col', question_col,
                        '--answer-col', answer_col,
                    ]
                    if verbose:
                        eval_cmd.append('--verbose')

                    result = subprocess.run(eval_cmd)
                    if result.returncode != 0:
                        log_step(f"QA evaluation failed for {model_name}", "error")
                        results[model_name] = {"qa": True, "evaluation": False, "error": "evaluation_failed"}
                        continue

                    results[model_name] = {"qa": True, "evaluation": True, "metrics_file": str(metrics_file)}

                log_step(f"Completed {model_name} successfully!", "success")

            except Exception as e:
                log_step(f"Error processing {config_file.name}: {e}", "error")
                # Use config filename as fallback if model_name is not defined
                fallback_name = config_file.stem if 'model_name' not in locals() else model_name
                results[fallback_name] = {"paraphrase": False, "qa": False, "evaluation": False, "error": str(e)}

        # Summary
        log_step("=" * 50, "info")
        log_step(f"SUMMARY ({task_type.upper()})", "start")
        log_step("=" * 50, "info")

        for model_name, status in results.items():
            if task_type == 'paraphrase':
                gen_status = "✅" if status.get("paraphrase") else "❌"
                eval_status = "✅" if status.get("evaluation") else "❌"
                log_step(f"{model_name}: Paraphrase {gen_status}, Evaluation {eval_status}")
            else:
                gen_status = "✅" if status.get("qa") else "❌"
                eval_status = "✅" if status.get("evaluation") else "❌"
                log_step(f"{model_name}: QA {gen_status}, Evaluation {eval_status}")

        all_success = all(
            status.get("paraphrase") and status.get("evaluation") if task_type == 'paraphrase'
            else status.get("qa") and status.get("evaluation")
            for v in results.values() for status in [v]
        )

        # Save summary
        summary_file = output_path / f"summary_{task_type}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        log_substep(f"Summary saved to: {summary_file}")

        log_step(f"Overall: {'✅ ALL SUCCESS' if all_success else '❌ SOME FAILED'}")

        if not all_success:
            sys.exit(1)

    except Exception as e:
        log_step(f"Fatal error: {str(e)}", "error")
        if verbose:
            click.echo(click.style("Stack trace:", fg="red"))
            click.echo(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    cli()
