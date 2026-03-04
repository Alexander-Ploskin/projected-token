#!/bin/bash
# Run QA generation for all models sequentially (without evaluation)
# Usage: ./scripts/run_qa_generation.sh [--input-path PATH] [--output-dir DIR]

set -e

# Default parameters
INPUT_PATH="${INPUT_PATH:-data/popqa.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qa_generation}"
TEXT_COL="${TEXT_COL:-s_wiki_content}"
BATCH_SIZE="${BATCH_SIZE:-5}"
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-path)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --text-col)
            TEXT_COL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
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
echo "QA Generation Runner (all models)"
echo "=============================================="
echo "Input Path: $INPUT_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Text Column: $TEXT_COL"
echo "Batch Size: $BATCH_SIZE"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

cd "$(dirname "$0")/.."

# Get timestamp for output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Find all config files
CONFIG_DIR="configs"
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory not found: $CONFIG_DIR"
    exit 1
fi

echo ""
echo "Running QA generation for all models..."
echo ""

for config_file in "$CONFIG_DIR"/*.yaml; do
    if [ ! -f "$config_file" ]; then
        continue
    fi

    config_name=$(basename "$config_file" .yaml)
    echo "----------------------------------------------"
    echo "Processing: $config_name"
    echo "----------------------------------------------"

    # Run QA generation
    python -m evaluation.cli qa \
        --config "$config_file" \
        --input-path "$INPUT_PATH" \
        --output-path "$OUTPUT_DIR/${config_name}_qa_${TIMESTAMP}.jsonl" \
        --text-col "$TEXT_COL" \
        --batch-size "$BATCH_SIZE" \
        $VERBOSE

    if [ $? -eq 0 ]; then
        echo "✅ Completed: $config_name"
    else
        echo "❌ Failed: $config_name"
    fi

    echo ""
done

echo "=============================================="
echo "QA generation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
