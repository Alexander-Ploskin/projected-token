
"""Run PISCO paraphrase experiment on PopQA dataset."""

import json
from typing import Any, Dict, List

from tqdm import tqdm
from evaluation.models.pisco import PiscoModel


def load_popqa_test(path: str) -> List[Dict[str, Any]]:
    """Load PopQA test dataset from JSONL file."""
    with open(path, "r", encoding="utf-8") as fp:
        return [json.loads(line.strip()) for line in fp if line.strip()]


def main() -> None:
    input_path = "/home/jovyan/rpt/results/simple_qwen_paraphrase_test.jsonl"
    output_path = "/home/jovyan/rpt/results/pisco_paraphrase_test.jsonl"

    print("Loading PopQA test dataset...")
    data = load_popqa_test(input_path)
    print(f"Loaded {len(data)} samples")

    print("Initializing PISCO model...")
    model = PiscoModel(
        model_name_or_path="naver/pisco-mistral",
        device="cuda:0",
        trust_remote_code=True,
    )
    print("Model initialized!")

    print(f"Processing samples and saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as out_fp:
        for i, item in enumerate(tqdm(data, desc="Paraphrasing")):
            doc = item.get("s_wiki_content", "")
            if not doc:
                print(f"Skipping item {i}: no s_wiki_content")
                continue

            try:
                result = model(
                    doc,
                    model.PARAPHRASE_DEFAULT_PROMPT_TEMPLATE,
                    {"max_new_tokens": 512},
                )
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue

            output_item = {
                "id": item.get("id"),
                "subj": item.get("subj"),
                "prop": item.get("prop"),
                "obj": item.get("obj"),
                "question": item.get("question"),
                "text": doc,
                "rephrased_text": result,
            }
            out_fp.write(json.dumps(output_item, ensure_ascii=False) + "\n")

    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    main()
