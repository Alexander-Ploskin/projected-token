from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import torch
import json
import logging
from pathlib import Path
import re
import string
from tqdm import tqdm

from src.models.factory import build_tokenizer, build_xrag_model
from src.config.schema import ExperimentConfig
from src.encoders.hf_encoder import HFMeanPoolRetriever


logger = logging.getLogger(__name__)

@dataclass
class EvalResult:
    id: str
    prediction: str
    metrics: dict[str, float]

class BaseEvalRunner(ABC):
    """Data-agnostic base for evaluation runners."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = cfg.eval.device
        # Map eval.torch_dtype too if needed
        eval_dtype_map = {
            'bf16': torch.bfloat16,
            'fp16': torch.float16, 
            'fp32': torch.float32
        }
        self.dtype = eval_dtype_map.get(cfg.eval.torch_dtype, torch.float32)
        self.max_prompt_chars = cfg.eval.max_prompt_chars

        self._build_model()
        self.model.eval()

    def _build_model(self):
        model_cfg = self.cfg.model
        retriever_cfg = self.cfg.retriever
        projector_cfg = self.cfg.projector

        self.tokenizer, xrag_token_id = build_tokenizer(
            model_cfg.model_name_or_path, 
            model_cfg.xrag_token
        )

        # --- retriever ---
        self.retriever = None
        retriever_embed_dim = 0
        if retriever_cfg.retriever_name_or_path:
            self.retriever = HFMeanPoolRetriever(
                retriever_cfg.retriever_name_or_path, 
                torch_dtype=torch.bfloat16
            )
            self.retriever_tokenizer = self.retriever.tokenizer
            retriever_embed_dim = self.retriever.embed_dim  # Exact dim!
            self.retriever.to(self.device)
            logger.info(f"Retriever loaded | embed_dim={retriever_embed_dim}")
        else:
            logger.warning("No retriever configured")
        
        if retriever_embed_dim == 0:
            raise ValueError("Need retriever for projector dim")

        # Map torch_dtype
        torch_dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
        torch_dtype = torch_dtype_map.get(model_cfg.torch_dtype, None)

        self.model = build_xrag_model(
            model_name_or_path=model_cfg.model_name_or_path,
            xrag_token_id=xrag_token_id,
            retriever_embed_dim=retriever_embed_dim,  # Now matches training!
            bridge_hidden_dim=projector_cfg.hidden_dim,
            bridge_dropout=projector_cfg.dropout,
            use_flash_attn_2=model_cfg.use_flash_attn_2,
            torch_dtype=torch_dtype,
        )

        # Load projector (dims now match exactly)
        ckpt = torch.load(projector_cfg.checkpoint, map_location='cpu')
        self.model.projector.load_state_dict(ckpt)
        self.model.freeze_llm()
        self.model.to(self.device)

    def generate(self, prompt: str, use_xrag: bool = True) -> str:
        """Generate using xRAG forward() loop."""
        gen_cfg = self.cfg.eval.generation
        max_new_tokens = gen_cfg["max_new_tokens"]
        
        # Build prompt with [XRAG] if needed
        prompts_dir = Path(self.cfg.eval.prompts_dir)
        if use_xrag:
            path = prompts_dir / self.cfg.eval.system_prompt_context
        else:
            path = prompts_dir / self.cfg.eval.system_prompt_basic
        
        with open(path, 'r') as f:
            system_prompt = f.read()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        if use_xrag:
            text += f" {self.cfg.model.xrag_token}"  # Use actual token
        
        if len(text) > self.max_prompt_chars:
            text = text[:self.max_prompt_chars]
        
        # Tokenize full prompt
        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Retrieval embeds (TODO: integrate HFMeanPoolRetriever)
        retrieval_embeds = None
        
        self.model.eval()
        with torch.no_grad():
            # Autoregressive generation loop
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    retrieval_embeds=retrieval_embeds,
                )
                next_token_logits = outputs.logits[:, -1, :]
                
                if gen_cfg["do_sample"]:
                    probs = torch.softmax(next_token_logits / gen_cfg["temperature"], dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # Append to inputs
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                
                # EOS stop
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated tokens only
        response = self.tokenizer.decode(
            input_ids[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        return response

    @abstractmethod
    def load_dataset(self) -> list[dict[str, Any]]:
        """Return [{'id': str, 'question': str, 'gold_answers': list[str]}]."""
        pass

    @abstractmethod
    def compute_row_metrics(self, prediction: str, gold_answers: list[str]) -> dict[str, float]:
        pass

    def normalize_answer(self, s: str) -> str:
        def remove_articles(text): 
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text): 
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        s = s.lower()
        s = remove_articles(s)
        s = remove_punc(s)
        s = white_space_fix(s)
        return s

    def run(self):
        dataset = self.load_dataset()
        results = []

        use_context = self.cfg.eval.use_context
        for item in tqdm(dataset, desc="Evaluating"):
            raw_pred = self.generate(item['question'], use_xrag=use_context)
            pred = self._clean_prediction(raw_pred)
            metrics = self.compute_row_metrics(pred, item['gold_answers'])

            results.append(EvalResult(item['id'], pred, metrics))

        aggregate_metrics = self._aggregate_metrics(results)
        self._save_results(results, aggregate_metrics)
        logger.info(f"Aggregate: {json.dumps(aggregate_metrics, indent=2)}")

    def _clean_prediction(self, text: str) -> str:
        text = text.strip()
        if text.endswith('.'): 
            text = text[:-1]
        if " is " in text: 
            text = text.split(" is ")[-1]
        elif " was " in text: 
            text = text.split(" was ")[-1]
        text = re.sub(r'^(an|a|the) ', '', text, flags=re.IGNORECASE)
        return text.strip()

    def _aggregate_metrics(self, results: list[EvalResult]) -> dict[str, float]:
        n = len(results)
        agg = {}
        for k in results[0].metrics:
            agg[k] = sum(r.metrics[k] for r in results) / n
        return agg

    def _save_results(self, results: list[EvalResult], agg_metrics: dict[str, float]):
        path = Path(self.cfg.eval.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        details = [{"id": r.id, "prediction": r.prediction, "metrics": r.metrics} for r in results]
        with open(path, 'w') as f:
            json.dump({
                "config": {
                    "task": self.cfg.task,
                    "model": self.cfg.model.model_name_or_path,
                    "use_context": self.cfg.eval.use_context
                },
                "aggregate_metrics": agg_metrics,
                "results": details[:100]
            }, f, indent=2)
