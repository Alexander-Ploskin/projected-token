from evaluation.metrics import Metric
from typing import Any, Dict, List, Optional
import json
import time
from enum import Enum
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field, confloat


class Label(str, Enum):
    supported = "supported"
    partially_supported = "partially_supported"
    contradicted = "contradicted"
    unknown = "unknown"


class FactualityVerdict(BaseModel):
    supported_claims: List[str] = Field(default_factory=list)
    contradicted_claims: List[str] = Field(default_factory=list)
    not_in_reference: List[str] = Field(default_factory=list)
    rationale: str = Field(description="Brief explanation; no long chain-of-thought.")
    score: confloat(ge=0.0, le=1.0) = Field(description="0..1 factual consistency of Text A relative to Text B")
    label: Label


class VerdictWithId(FactualityVerdict):
    id: int = Field(description="Must match the input case id.")


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
- If A adds details not present in B, list them under not_in_reference (do NOT assume).
- If A conflicts with B, list conflicts under contradicted_claims.
- Keep claims short and atomic when listing.
- Return one verdict per input case.

Return only the structured output with:
{ "verdicts": [ ... ] }
"""


class GPTScoreMetric(Metric):
    def __init__(self, config: dict) -> None:
        """
        Initialize LLM Judge metric.
        
        Expected config parameters:
        - base_url: API base URL (e.g., "http://localhost:8000/v1")
        - api_key: API key
        - model: model name (e.g., "openai/gpt-oss-20b")
        - temperature: sampling temperature
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
        Calculate factual consistency score between original and rephrased text using LLM judge.
        
        Args:
            original: Original reference text
            rephrased: Rephrased candidate text to evaluate
            
        Returns:
            Dictionary with judge score, label, and detailed claims
        """
        try:
            # Create a single case for the judge
            cases = [{"id": 0, "candidate": rephrased, "reference": original}]
            
            # Use the batch interface but with just one item
            response: BatchVerdicts = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_model=BatchVerdicts,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({"cases": cases}, ensure_ascii=False)},
                ],
            )
            
            verdict = response.verdicts[0]  # Get the first (and only) verdict
            
            return {
                "llm_judge_score": float(verdict.score),
                "llm_judge_label": verdict.label.value,
                "llm_judge_supported_claims": verdict.supported_claims,
                "llm_judge_contradicted_claims": verdict.contradicted_claims,
                "llm_judge_not_in_reference": verdict.not_in_reference,
                "llm_judge_rationale": verdict.rationale
            }
            
        except Exception as e:
            # Return default values on error
            return {
                "llm_judge_score": -1.0,
                "llm_judge_label": "error",
                "llm_judge_supported_claims": [],
                "llm_judge_contradicted_claims": [],
                "llm_judge_not_in_reference": [],
                "llm_judge_rationale": f"Error: {str(e)}"
            }

    def judge_batch(self, candidates: List[str], references: List[str]) -> List[Dict[str, Any]]:
        """
        Judge a batch of candidate-reference pairs and return full verdict details
        
        Args:
            candidates: List of candidate texts
            references: List of reference texts
            
        Returns:
            List of verdict dictionaries with full details
        """
        if len(candidates) != len(references):
            raise ValueError("Candidates and references must have the same length")
        
        # Create cases with IDs
        cases = [{"id": i, "candidate": cand, "reference": ref} 
                for i, (cand, ref) in enumerate(zip(candidates, references))]
        
        try:
            response: BatchVerdicts = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_model=BatchVerdicts,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({"cases": cases}, ensure_ascii=False)},
                ],
            )
            
            # Convert to list of verdict dictionaries
            verdicts = []
            for verdict in response.verdicts:
                verdicts.append({
                    "llm_judge_score": float(verdict.score),
                    "llm_judge_label": verdict.label.value,
                    "llm_judge_supported_claims": verdict.supported_claims,
                    "llm_judge_contradicted_claims": verdict.contradicted_claims,
                    "llm_judge_not_in_reference": verdict.not_in_reference,
                    "llm_judge_rationale": verdict.rationale
                })
            
            return verdicts
            
        except Exception as e:
            # Return error verdicts for all items
            return [{
                "llm_judge_score": -1.0,
                "llm_judge_label": "error",
                "llm_judge_supported_claims": [],
                "llm_judge_contradicted_claims": [],
                "llm_judge_not_in_reference": [],
                "llm_judge_rationale": f"Batch error: {str(e)}"
            } for _ in range(len(candidates))]
