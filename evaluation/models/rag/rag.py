import torch
from transformers import AutoTokenizer
from typing import List, Optional, Dict

from evaluation.models import Model
from evaluation.models.xrag.xmistral import XMistralForCausalLM


class RAGModel(Model):
    QA_PROMPT_TEMPLATE = (
"""
Answer the Question concisely and completely, with no filler or explanation.
Document: {document}
Question: {question}
Answer:
"""
)
    PARAPHRASE_PROMPT_TEMPLATE = "Background: {document}\n Provide a paraphrase of the background sentences (background document context). Provide only the paraphrased text, no other text or formatting. Keep the meaning and all facts intact."
    PARAPHRASE_DEFAULT_PROMPT_TEMPLATE = PARAPHRASE_PROMPT_TEMPLATE  # Alias for CLI
    default_prompt_template = QA_PROMPT_TEMPLATE

    def __init__(self, llm_name_or_path: str, device: str) -> None:
        self._device = torch.device(device)
        llm_name_or_path = llm_name_or_path
        self._llm = XMistralForCausalLM.from_pretrained(llm_name_or_path,torch_dtype = torch.bfloat16,low_cpu_mem_usage = True,).to(self._device).eval()
        self._llm_tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path,add_eos_token=False,use_fast=False,padding_side='left')

    def __call__(self, document: str, prompt_template: str, model_args: dict = {}) -> str:
        return self.generate_batch([document], prompt_template, model_args)[0]

    def generate_batch(
        self,
        documents: List[str],
        prompt_template: Optional[str] = None,
        model_args: Optional[Dict] = None,
        questions: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate answers for a batch of documents.

        Args:
            documents: List of document texts to process (or pre-formatted prompts for QA)
            prompt_template: Template with {document} and optionally {question} placeholders
            model_args: Generation arguments (do_sample, temperature, max_new_tokens, etc.)
            questions: Optional list of questions. If provided, used with {question} in template.

        Returns:
            List of generated answers
        """
        if model_args is None:
            model_args = {}
        if prompt_template is None:
            prompt_template = self.default_prompt_template

        # Format prompts based on available parameters
        if questions is not None:
            # QA mode: format with both document and question
            prompts = [
                prompt_template.format(document=doc, question=question)
                for doc, question in zip(documents, questions)
            ]
        elif prompt_template and prompt_template.strip():
            # Paraphrase mode: format with document only
            prompts = [prompt_template.format_map(dict(document=doc)) for doc in documents]
        else:
            # Documents are already formatted prompts
            prompts = documents

        # Tokenize batch with padding
        encoded = self._llm_tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
        )
        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        # Batch generation
        with torch.inference_mode():
            generated_output = self._llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self._llm_tokenizer.pad_token_id,
                **model_args
            )

        # Decode outputs, skipping input tokens
        results = []
        for i, gen in enumerate(generated_output):
            start_idx = input_ids[i].shape[0]
            decoded = self._llm_tokenizer.batch_decode(
                gen[start_idx:],
                skip_special_tokens=True
            )[0]
            results.append(decoded)

        return results
