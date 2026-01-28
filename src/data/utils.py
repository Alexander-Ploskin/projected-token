from __future__ import annotations

import logging
from typing import Optional
from datasets import load_dataset, DatasetDict, Dataset, IterableDataset
from datasets.exceptions import DatasetNotFoundError

logger = logging.getLogger(__name__)


def load_json_dataset(train_file: str, dev_file: str | None) -> DatasetDict:
    """
    Load dataset from local JSONl files.
    
    Args:
        train_file: Path to training JSONl file
        dev_file: Optional path to dev/validation JSONl file
        
    Returns:
        DatasetDict with 'train' and optionally 'dev' splits
    """
    data_files = {"train": train_file}
    if dev_file:
        data_files["dev"] = dev_file
    
    logger.info(f"Loading JSONl dataset from: {train_file}")
    if dev_file:
        logger.info(f"Dev set: {dev_file}")
    
    return load_dataset("json", data_files=data_files)


def load_huggingface_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split_train: str = "train",
    split_dev: Optional[str] = "validation",
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> DatasetDict:
    """
    Load dataset from HuggingFace Hub.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "HuggingFaceFW/finewiki")
        subset: Dataset configuration/subset (e.g., "en" for finewiki)
        split_train: HF dataset split name for training (default: "train")
        split_dev: HF dataset split name for dev/validation (default: "validation")
        streaming: Whether to use streaming mode for large datasets
        cache_dir: Custom cache directory for downloaded datasets
        revision: Pin specific dataset version/commit for reproducibility
        
    Returns:
        DatasetDict with 'train' and optionally 'dev' splits
        
    Raises:
        DatasetNotFoundError: If dataset not found on HuggingFace Hub
        ValueError: If required splits are missing
    """
    logger.info(f"Loading HuggingFace dataset: {dataset_name}")
    if subset:
        logger.info(f"  Subset/config: {subset}")
    if revision:
        logger.info(f"  Revision: {revision}")
    logger.info(f"  Streaming mode: {streaming}")
    
    try:
        # Load the full dataset to inspect available splits
        dataset = load_dataset(
            dataset_name,
            name=subset,
            streaming=streaming,
            cache_dir=cache_dir,
            revision=revision,
        )
        
        # Check if dataset is already a DatasetDict or needs split handling
        if isinstance(dataset, (DatasetDict, dict)):
            available_splits = list(dataset.keys())
            logger.info(f"  Available splits: {available_splits}")
            
            # Validate and map splits
            result = {}
            
            # Handle training split
            if split_train in available_splits:
                result["train"] = dataset[split_train]
                logger.info(f"  Using '{split_train}' split for training")
            else:
                raise ValueError(
                    f"Training split '{split_train}' not found. "
                    f"Available splits: {available_splits}. "
                    f"Please set dataset_split_train to one of the available splits."
                )
            
            # Handle dev/validation split
            if split_dev:
                if split_dev in available_splits:
                    result["dev"] = dataset[split_dev]
                    logger.info(f"  Using '{split_dev}' split for validation")
                else:
                    logger.warning(
                        f"Dev split '{split_dev}' not found in {available_splits}. "
                    )
                    if streaming:
                        logger.info("  Streaming mode: Creating 'dev' split by taking first 1000 samples.")
                        # For streaming, we take some samples for dev and skip them for train
                        result["dev"] = dataset[split_train].take(1000)
                        result["train"] = dataset[split_train].skip(1000)
                    else:
                        logger.info("  Non-streaming: Splitting training data (1% for validation, max 1000 samples).")
                        # For non-streaming, use train_test_split
                        train_size = len(dataset[split_train])
                        test_size = min(1000, max(1, int(train_size * 0.01)))
                        split_data = dataset[split_train].train_test_split(test_size=test_size, seed=42)
                        result["train"] = split_data["train"]
                        result["dev"] = split_data["test"]
                    
                    logger.info(f"  Created 'dev' split from training data.")
            
            return DatasetDict(result)
        
        elif isinstance(dataset, (Dataset, IterableDataset)):
            # Single split dataset, use as training data
            logger.warning(
                f"Dataset has no splits, using entire dataset for training"
            )
            return DatasetDict({"train": dataset})
        
        else:
            raise ValueError(f"Unexpected dataset type: {type(dataset)}")
            
    except DatasetNotFoundError as e:
        raise DatasetNotFoundError(
            f"Dataset '{dataset_name}' not found on HuggingFace Hub. "
            f"Please check the dataset name and subset. "
            f"You can browse available datasets at https://huggingface.co/datasets"
        ) from e
    except Exception as e:
        logger.error(f"Error loading HuggingFace dataset: {e}")
        raise


def load_dataset_auto(
    train_file: Optional[str] = None,
    dev_file: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_subset: Optional[str] = None,
    dataset_split_train: str = "train",
    dataset_split_dev: Optional[str] = "validation",
    streaming: bool = False,
    dataset_cache_dir: Optional[str] = None,
    dataset_revision: Optional[str] = None,
) -> DatasetDict:
    """
    Smart dispatcher that auto-detects data source and loads accordingly.
    
    Priority:
    1. If dataset_name is provided, use HuggingFace loader
    2. Otherwise, use JSONl file loader with train_file
    
    Args:
        train_file: Path to training JSONl file (for local loading)
        dev_file: Path to dev JSONl file (for local loading)
        dataset_name: HuggingFace dataset identifier
        dataset_subset: Dataset configuration/subset
        dataset_split_train: HF split name for training
        dataset_split_dev: HF split name for dev/validation
        streaming: Use streaming mode for HF datasets
        dataset_cache_dir: Custom cache directory
        dataset_revision: Pin dataset version
        
    Returns:
        DatasetDict with 'train' and optionally 'dev' splits
        
    Raises:
        ValueError: If no data source is configured
    """
    # Option 1: Load from HuggingFace Hub
    if dataset_name:
        logger.info("=" * 60)
        logger.info("Data Source: HuggingFace Hub")
        logger.info("=" * 60)
        return load_huggingface_dataset(
            dataset_name=dataset_name,
            subset=dataset_subset,
            split_train=dataset_split_train,
            split_dev=dataset_split_dev,
            streaming=streaming,
            cache_dir=dataset_cache_dir,
            revision=dataset_revision,
        )
    
    # Option 2: Load from local JSONl files
    elif train_file:
        logger.info("=" * 60)
        logger.info("Data Source: Local JSONl Files")
        logger.info("=" * 60)
        return load_json_dataset(train_file, dev_file)
    
    # No data source configured
    else:
        raise ValueError(
            "No data source configured. Please provide either:\n"
            "  - dataset_name for HuggingFace datasets, or\n"
            "  - train_file for local JSONl files"
        )


def validate_dataset_structure(
    dataset: DatasetDict,
    required_fields: Optional[list[str]] = None,
    task: str = "pretrain",
) -> None:
    """
    Validate dataset structure and schema.
    
    Args:
        dataset: DatasetDict to validate
        required_fields: List of required field names (if None, use task defaults)
        task: Task type (used to determine default required fields)
        
    Raises:
        ValueError: If dataset structure is invalid or missing required fields
    """
    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"Expected DatasetDict, got {type(dataset)}")
    
    if "train" not in dataset:
        raise ValueError("Dataset must have a 'train' split")
    
    # Get first split to check schema
    train_split = dataset["train"]
    
    # Handle streaming datasets differently
    if isinstance(train_split, IterableDataset):
        logger.info("Streaming dataset detected - skipping detailed schema validation")
        logger.info("  Schema will be validated during preprocessing")
        return
    
    # Check dataset is not empty
    if len(train_split) == 0:
        raise ValueError("Training split is empty")
    
    logger.info(f"Dataset structure:")
    logger.info(f"  Splits: {list(dataset.keys())}")
    logger.info(f"  Train samples: {len(train_split)}")
    if "dev" in dataset:
        logger.info(f"  Dev samples: {len(dataset['dev'])}")
    
    # Get features/columns
    features = train_split.features if hasattr(train_split, 'features') else train_split.column_names
    logger.info(f"  Features: {list(features) if isinstance(features, dict) else features}")
    
    # Validate required fields if specified
    if required_fields:
        if isinstance(features, dict):
            available_fields = set(features.keys())
        else:
            available_fields = set(features)
        
        missing_fields = set(required_fields) - available_fields
        if missing_fields:
            raise ValueError(
                f"Dataset missing required fields: {missing_fields}\n"
                f"Available fields: {available_fields}"
            )
        logger.info(f"  ✓ All required fields present: {required_fields}")
    
    # Show sample data for inspection
    if hasattr(train_split, '__getitem__'):
        sample = train_split[0]
        logger.info(f"  Sample keys: {list(sample.keys())}")
        logger.info(f"  Sample preview: {str(sample)[:200]}...")

