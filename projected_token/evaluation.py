import os
import re
import logging
import string
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
from jinja2 import Template  # Added for template rendering


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PopQAEvaluator:
    """
    Evaluates a Language Model on the PopQA dataset.
    """

    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(self, model_id: str = None, device: str = "cuda"):
        """
        Initialize the evaluator with model and tokenizer.
        """
        self.model_id = model_id or self.MODEL_ID
        self.device = device if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading model: {self.model_id} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        if self.device == "cpu":
            self.model.to(self.device)
            
        self.model.eval()
        logger.info("Model loaded successfully.")

    def clean_generation(self, text: str) -> str:
        """
        Aggressively cleans the output if the model fails to be concise.
        """
        text = text.strip()
        if text.endswith('.'):
            text = text[:-1]
        
        if " is " in text:
            text = text.split(" is ")[-1]
        elif " was " in text:
            text = text.split(" was ")[-1]
            
        text = re.sub(r'^(an|a|the) ', '', text, flags=re.IGNORECASE)
        
        return text.strip()

    def generate_answer(self, question: str, context: Optional[str] = None) -> str:
        """
        Generates an answer for a single question using the model.
        """
        prompt_folder = '/workspace/prompts/'
        
        if context:
            prompt_path = os.path.join(prompt_folder, 'context.hbs')
            with open(prompt_path, 'r') as fp:
                template_content = fp.read()
            
            # Render the template with the context
            template = Template(template_content)
            system_prompt = template.render(context=context)
            
        else:
            prompt_path = os.path.join(prompt_folder, 'basic.txt')
            with open(prompt_path, 'r') as fp:
                system_prompt = fp.read()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Truncate roughly to avoid OOM (safeguard)
        if len(text) > 12000: 
            text = text[:12000]
            
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=30,
                do_sample=True,     
                temperature=0.1,
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    @staticmethod
    def normalize_answer(s: str) -> str:
        """Standard SQuAD normalization."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_metrics(self, ground_truths: List[str], prediction: str) -> Dict[str, float]:
        """
        Computes exact match, f1, and InAcc.
        """
        normalized_pred = self.normalize_answer(prediction)
        normalized_gts = [self.normalize_answer(gt) for gt in ground_truths]
        
        em = max([int(normalized_pred == norm_gt) for norm_gt in normalized_gts]) if normalized_gts else 0
        
        f1_scores = []
        pred_tokens = normalized_pred.split()
        for norm_gt in normalized_gts:
            gt_tokens = norm_gt.split()
            if len(pred_tokens) == 0 or len(gt_tokens) == 0:
                f1_scores.append(int(pred_tokens == gt_tokens))
                continue
            
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                f1_scores.append(0)
                continue
            precision = 1.0 * num_same / len(pred_tokens)
            recall = 1.0 * num_same / len(gt_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1)
        f1 = max(f1_scores) if f1_scores else 0.0

        in_acc = max([int(norm_gt in normalized_pred) for norm_gt in normalized_gts]) if normalized_gts else 0

        return {"em": em, "f1": f1, "in_acc": in_acc}

    def evaluate(self, input_file: str, output_file: str, limit: int = None, use_context: bool = False):
        """
        Main evaluation loop.
        """
        if input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
        else:
            try:
                df = pd.read_csv(input_file, sep='\t')
            except:
                df = pd.read_json(input_file, lines=True)

        if limit:
            df = df.head(limit)

        logger.info(f"Starting evaluation on {len(df)} samples (Use Context: {use_context})...")
        
        results = []
        metrics_summary = {"em": 0, "f1": 0, "in_acc": 0}

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            question = row['question']
            
            context_text = None
            if use_context:
                contents = []
                if pd.notna(row.get('s_wiki_content')):
                    contents.append(str(row['s_wiki_content']))
                if pd.notna(row.get('o_wiki_content')):
                    contents.append(str(row['o_wiki_content']))
                
                if contents:
                    context_text = "\n\n".join(contents)
                    if len(context_text) > 15000:
                        context_text = context_text[:15000] + "...(truncated)"
            
            possible_answers = row['possible_answers']
            if isinstance(possible_answers, str):
                try:
                    if possible_answers.strip().startswith("["):
                        possible_answers = eval(possible_answers)
                    else:
                        possible_answers = [possible_answers]
                except:
                    possible_answers = [possible_answers]
            elif not isinstance(possible_answers, list):
                possible_answers = [str(possible_answers)]
            
            raw_prediction = self.generate_answer(question, context=context_text)
            prediction = self.clean_generation(raw_prediction)
            
            row_metrics = self.compute_metrics(possible_answers, prediction)
            
            metrics_summary["em"] += row_metrics["em"]
            metrics_summary["f1"] += row_metrics["f1"]
            metrics_summary["in_acc"] += row_metrics["in_acc"]
            
            results.append({
                "id": row.get('id', idx),
                "question": question,
                "gold_answers": possible_answers,
                "context_used": bool(context_text),
                "prediction": prediction,
                "metrics": row_metrics
            })

        total = len(df) or 1
        final_metrics = {k: v / total for k, v in metrics_summary.items()}
        
        logger.info("Evaluation Complete.")
        logger.info(f"Final Metrics: {json.dumps(final_metrics, indent=2)}")
        
        with open(output_file, 'w') as f:
            json.dump({"config": {"model": self.model_id, "use_context": use_context}, 
                       "metrics": final_metrics, 
                       "details": results}, f, indent=2)
        logger.info(f"Detailed results saved to {output_file}")
