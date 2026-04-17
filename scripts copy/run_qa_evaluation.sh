#!/bin/bash
# Run QA evaluation for all generated answers
# Usage: ./scripts/run_qa_evaluation.sh [--input-dir DIR] [--output-dir DIR]

set -e

# Default parameters
GPT_BASE_URL="${GPT_BASE_URL:-http://localhost:8000/v1}"
GPT_MODEL="${GPT_MODEL:-Qwen/Qwen3.5-27B}"
GPT_API_KEY="${GPT_API_KEY:-dummy}"
GPT_BATCH_SIZE="${GPT_BATCH_SIZE:-10}"
INPUT_DIR="${INPUT_DIR:-results/qa_generation}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qa_evaluation}"
QUESTION_COL="${QUESTION_COL:-question}"
ANSWER_COL="${ANSWER_COL:-possible_answers}"
PREDICTED_COL="${PREDICTED_COL:-answer}"
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
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --question-col)
            QUESTION_COL="$2"
            shift 2
            ;;
        --answer-col)
            ANSWER_COL="$2"
            shift 2
            ;;
        --predicted-col)
            PREDICTED_COL="$2"
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
echo "QA Evaluation Runner (all models)"
echo "=============================================="
echo "GPT Base URL: $GPT_BASE_URL"
echo "GPT Model: $GPT_MODEL"
echo "GPT Batch Size: $GPT_BATCH_SIZE"
echo "Input Dir: $INPUT_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

cd "$(dirname "$0")/.."

echo ""
echo "Running QA evaluation for all generated answers..."
echo ""

# Find all JSONL files in input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

for input_file in "$INPUT_DIR"/*.jsonl; do
    if [ ! -f "$input_file" ]; then
        continue
    fi

    input_name=$(basename "$input_file" .jsonl)
    output_file="$OUTPUT_DIR/${input_name}_metrics.json"

    echo "----------------------------------------------"
    echo "Evaluating: $input_name"
    echo "----------------------------------------------"

    # Run QA evaluation
    python -m evaluation.cli eval-qa \
        --input-path "$input_file" \
        --output "$output_file" \
        --base-url "$GPT_BASE_URL" \
        --api-key "$GPT_API_KEY" \
        --model "$GPT_MODEL" \
        --batch-size "$GPT_BATCH_SIZE" \
        --question-col "$QUESTION_COL" \
        --answer-col "$ANSWER_COL" \
        --predicted-col "$PREDICTED_COL" \
        $VERBOSE

    if [ $? -eq 0 ]; then
        echo "✅ Completed: $input_name"
        echo "   Metrics saved to: $output_file"

        # Print summary metrics
        if command -v python &> /dev/null; then
            python -c "
import json
with open('$output_file') as f:
    metrics = json.load(f)
print(f'   avg_qa_score: {metrics.get(\"avg_qa_score\", \"N/A\")}')
print(f'   in_accuracy: {metrics.get(\"in_accuracy\", \"N/A\")}')
"
        fi
    else
        echo "❌ Failed: $input_name"
    fi

    echo ""
done

echo "=============================================="
echo "QA evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
