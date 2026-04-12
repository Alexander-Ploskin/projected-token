#!/bin/bash
# Run QA experiments for all models sequentially
# Usage: ./scripts/run_qa_all.sh [--gpt-base-url URL] [--gpt-model MODEL]

set -e

# Default parameters
GPT_BASE_URL="${GPT_BASE_URL:-http://localhost:8000/v1}"
GPT_MODEL="${GPT_MODEL:-Qwen/Qwen3.5-27B}"
GPT_API_KEY="${GPT_API_KEY:-dummy}"
GPT_BATCH_SIZE="${GPT_BATCH_SIZE:-10}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-5}"
INPUT_PATH="${INPUT_PATH:-data/popqa.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qa_experiments}"
TEXT_COL="${TEXT_COL:-s_wiki_content}"
QUESTION_COL="${QUESTION_COL:-question}"
ANSWER_COL="${ANSWER_COL:-possible_answers}"
TASK_TYPE="qa"
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpt-base-url)
            GPT_BASE_URL="$2"
            shift 2
            ;;
        --gpt-model)
            GPT_MODEL="$2"
            shift 2
            ;;
        --gpt-api-key)
            GPT_API_KEY="$2"
            shift 2
            ;;
        --gpt-batch-size)
            GPT_BATCH_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            INFERENCE_BATCH_SIZE="$2"
            shift 2
            ;;
        --input-path)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="-v"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "QA Experiment Runner"
echo "=============================================="
echo "GPT Base URL: $GPT_BASE_URL"
echo "GPT Model: $GPT_MODEL"
echo "GPT Batch Size: $GPT_BATCH_SIZE"
echo "Inference Batch Size: $INFERENCE_BATCH_SIZE"
echo "Input Path: $INPUT_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Task Type: $TASK_TYPE"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run all QA experiments using the CLI
cd "$(dirname "$0")/.."

echo ""
echo "Starting QA experiments..."
echo ""

python -m evaluation.cli run-all \
    --configs-dir configs \
    --input-path "$INPUT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --text-col "$TEXT_COL" \
    --question-col "$QUESTION_COL" \
    --answer-col "$ANSWER_COL" \
    --batch-size "$INFERENCE_BATCH_SIZE" \
    --gpt-base-url "$GPT_BASE_URL" \
    --gpt-api-key "$GPT_API_KEY" \
    --gpt-model "$GPT_MODEL" \
    --gpt-batch-size "$GPT_BATCH_SIZE" \
    --task-type "$TASK_TYPE" \
    $VERBOSE

echo ""
echo "=============================================="
echo "All QA experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
