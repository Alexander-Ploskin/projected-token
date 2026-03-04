"""
QA scoring metrics for question-answering evaluation following OSCAR paper (arxiv:2504.07109).

This module implements:
1. LLM-based QA evaluation (Appendix E): Scores answers as 1 (correct), 0.5 (partially correct), 0 (wrong)
2. in_accuracy metric (Appendix A.2): Checks if ground truth answer is contained in prediction
"""

import re
import json
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, confloat

from openai import OpenAI
import instructor

from evaluation.metrics import Metric


class QALabel(str, Enum):
    """Labels for QA evaluation following OSCAR paper."""
    correct = "correct"
    partially_correct = "partially_correct"
    wrong = "wrong"


class QAVerdict(BaseModel):
    """Structured output for QA evaluation verdict."""
    label: QALabel = Field(description="One of: correct, partially_correct, wrong")
    score: confloat(ge=0.0, le=1.0) = Field(description="1.0 for correct, 0.5 for partially correct, 0.0 for wrong")
    rationale: str = Field(description="Brief explanation for the verdict; no long chain-of-thought.")


class VerdictWithId(QAVerdict):
    """QA verdict with ID for batch processing."""
    id: int = Field(description="Must match the input case id.")


class BatchQAVerdicts(BaseModel):
    """Batch of QA verdicts."""
    verdicts: List[VerdictWithId]


# System prompt for QA evaluation (from OSCAR paper Appendix E, Figure 13)
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
- Compare the ai_answer against the golden_answer to determine correctness.

Rubric:
- correct (1): The AI answer is factually correct and matches the golden answer.
- partially_correct (0.5): The AI answer is partially correct or contains some correct information but is incomplete or has minor errors.
- wrong (0): The AI answer is factually incorrect or does not answer the question properly.

Guidelines:
- Treat the golden_answer as the ground truth.
- Consider semantic equivalence, not just exact string matching.
- If the AI answer contains the correct information but is verbose, it can still be correct.
- If the AI answer is completely off-topic or factually wrong, mark it as wrong.

Return only the structured output with:
{ "verdicts": [ ... ] }"""


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison following OSCAR paper's approach.

    - Lowercase
    - Remove punctuation
    - Normalize whitespace

    Args:
        text: Input text to normalize

    Returns:
        Normalized text string
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def compute_in_accuracy(predicted_answer: str, reference_answers: List[str]) -> bool:
    """
    Compute in_accuracy: checks if any ground truth answer is contained in the prediction.

    As described in OSCAR paper Appendix A.2, accuracy is 1 if the ground truth label
    is included as a substring of the generated answer after normalization.

    Args:
        predicted_answer: The predicted/generated answer
        reference_answers: List of ground truth answers

    Returns:
        True if any reference answer is a substring of the predicted answer
    """
    predicted_normalized = normalize_text(predicted_answer)
    for ref in reference_answers:
        ref_normalized = normalize_text(ref)
        if ref_normalized in predicted_normalized:
            return True
    return False


class QAScoreMetric(Metric):
    """
    LLM-based QA evaluation following OSCAR paper (Appendix E).

    Uses an LLM to judge if an answer is correct given a question and reference answer.
    Returns score: 1 (correct), 0.5 (partially correct), 0 (wrong)

    Also computes in_accuracy metric from Appendix A.2.

    Expected config parameters:
    - base_url: API base URL (e.g., "http://localhost:8000/v1")
    - api_key: API key
    - model: model name (e.g., "Qwen/Qwen3.5-27B")
    - temperature: sampling temperature (default: 0.0)
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize QA scoring metric.

        Args:
            config: Configuration dictionary with API and model settings
        """
        self.base_url = config["base_url"]
        self.api_key = config["api_key"]
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.0)

        # Initialize client
        raw_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.client = instructor.patch(raw_client, mode=instructor.Mode.JSON)

    def __call__(self, original: str, rephrased: str) -> dict[str, Any]:
        """
        Calculate QA score for a single question-answer pair.

        Note: This method signature follows the Metric base class interface.
        For QA evaluation, we interpret:
        - original: JSON string containing question and reference answer(s)
        - rephrased: The predicted answer to evaluate

        Args:
            original: JSON string with 'question' and 'reference_answers' keys
            rephrased: The predicted answer to evaluate

        Returns:
            Dictionary with qa_score, in_accuracy, label, and rationale
        """
        try:
            # Parse the original to get question and reference answers
            if isinstance(original, str):
                qa_data = json.loads(original)
            else:
                qa_data = original

            question = qa_data.get('question', '')
            reference_answers = qa_data.get('reference_answers', [])

            if not reference_answers:
                # Fallback: treat as single reference
                reference_answers = [qa_data.get('reference_answer', '')]

            # For LLM evaluation, join reference answers with " OR " if multiple
            golden_answer = " OR ".join(reference_answers)

            # Create a single case for the judge
            cases = [{
                "id": 0,
                "question": question,
                "golden_answer": golden_answer,
                "ai_answer": rephrased
            }]

            # Use the batch interface but with just one item
            response: BatchQAVerdicts = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_model=BatchQAVerdicts,
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({"cases": cases}, ensure_ascii=False)},
                ],
            )

            verdict = response.verdicts[0]  # Get the first (and only) verdict

            # Compute in_accuracy
            in_acc = compute_in_accuracy(rephrased, reference_answers)

            return {
                "qa_score": float(verdict.score),
                "in_accuracy": in_acc,
                "label": verdict.label.value,
                "rationale": verdict.rationale
            }

        except Exception as e:
            # Return default values on error
            return {
                "qa_score": -1.0,
                "in_accuracy": False,
                "label": "error",
                "rationale": f"Error: {str(e)}"
            }

    def judge_batch(self, questions: List[str], reference_answers_list: List[List[str]],
                   predicted_answers: List[str]) -> List[Dict[str, Any]]:
        """
        Judge a batch of question-answer pairs and return full verdict details.

        Args:
            questions: List of questions
            reference_answers_list: List of lists of reference answers (one list per question)
            predicted_answers: List of predicted answers

        Returns:
            List of verdict dictionaries with full details
        """
        if len(questions) != len(reference_answers_list) or len(questions) != len(predicted_answers):
            raise ValueError("Questions, reference answers, and predicted answers must have the same length")

        # Create cases with IDs
        cases = []
        for i, (question, ref_answers, pred_answer) in enumerate(zip(questions, reference_answers_list, predicted_answers)):
            golden_answer = " OR ".join(ref_answers)
            cases.append({
                "id": i,
                "question": question,
                "golden_answer": golden_answer,
                "ai_answer": pred_answer
            })

        try:
            response: BatchQAVerdicts = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_model=BatchQAVerdicts,
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({"cases": cases}, ensure_ascii=False)},
                ],
            )

            # Convert to list of verdict dictionaries
            verdicts = []
            for i, verdict in enumerate(response.verdicts):
                # Compute in_accuracy for this item
                in_acc = compute_in_accuracy(predicted_answers[i], reference_answers_list[i])

                verdicts.append({
                    "qa_score": float(verdict.score),
                    "in_accuracy": in_acc,
                    "label": verdict.label.value,
                    "rationale": verdict.rationale
                })

            return verdicts

        except Exception as e:
            # Return error verdicts for all items
            return [{
                "qa_score": -1.0,
                "in_accuracy": False,
                "label": "error",
                "rationale": f"Batch error: {str(e)}"
            } for _ in range(len(predicted_answers))]
