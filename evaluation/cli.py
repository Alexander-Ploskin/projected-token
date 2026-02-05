import click
import yaml
from typing import Dict, Any
import json
import sys
import traceback
from datetime import datetime
import random


PROMPT_TEMPLATE = "Background: {document} Could you give me a different version of the background sentences above?"

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
    """CLI for paraphrasing and evaluation tasks."""
    pass

@cli.command()
@click.option('--config', required=True, help='Path to YAML configuration file')
@click.option('--input-path', required=True, help='Input dataset path')
@click.option('--output-path', required=True, help='Output file path')
@click.option('--text-col', default='text', help='Column name for input text')
@click.option('--output-col', default='rephrased_text', help='Column name for rephrased text')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def paraphrase(config, input_path, output_path, text_col, output_col, verbose):
    """Rephrase text using a model and save results."""
    log_step("Starting paraphrase operation", "start")
    
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

        # Get model generation parameters from config
        model_args = config_data.get('generation', {})
        log_substep(f"Generation parameters: {model_args}")

        experiment_config = config_data.get('experiment', {})
        log_substep(f"Experiment parameters: {experiment_config}")

        prompt_template = (
            experiment_config.get('prompt_template')
            or getattr(model, 'default_prompt_template', None)
            or PROMPT_TEMPLATE
        )
        log_substep(f"Prompt template: `{prompt_template}`")

        # Process data
        log_step("Processing data")
        output_data = []
        processed_count = 0
        error_count = 0
        
        for i, item in enumerate(dataset_instance):
            if verbose and i % 100 == 0:
                log_substep(f"Processed {i} items...")
            
            document = item.get(text_col, '')
            
            if not document:
                if verbose:
                    log_step(f"Skipping item {i}: missing text column '{text_col}'", "warning")
                error_count += 1
                continue
            
            try:
                for _ in range(int(experiment_config.get('sample_count', 1))):
                    rephrased = model(document, prompt_template, model_args)
                    output_data.append(item | {output_col: rephrased})
                    processed_count += 1
                    
                    if verbose and processed_count % 10 == 0:
                        log_substep(f"Sample rephrasing: {rephrased[:100]}...")
                    
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
        
        log_step(f"Paraphrasing completed successfully!", "success")
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