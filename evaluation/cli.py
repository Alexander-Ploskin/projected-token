import click
import yaml
from typing import Dict, Any, List, Optional
import json
import sys
import traceback
from datetime import datetime
from tqdm import tqdm

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

if __name__ == '__main__':
    cli()
