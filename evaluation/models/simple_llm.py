import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict

from evaluation.models import Model


class SimpleLLM(Model):
    """Simple LLM wrapper for standard causal language models (Qwen, Llama, etc.)"""

    QA_PROMPT_TEMPLATE = (
"""Answer the Question concisely and completely, with no filler or explanation.
Document: {document}
Question: {question}
Answer:
"""
)
    PARAPHRASE_PROMPT_TEMPLATE = (
"""Paraphrase the following text. Keep the meaning and all facts intact, but use different wording:
{document}"""
)
    PARAPHRASE_DEFAULT_PROMPT_TEMPLATE = PARAPHRASE_PROMPT_TEMPLATE
    QA_DEFAULT_PROMPT_TEMPLATE = QA_PROMPT_TEMPLATE
    default_prompt_template = QA_PROMPT_TEMPLATE

    def __init__(self, **kwargs) -> None:
        self._device = torch.device(kwargs["device"])
        model_name_or_path = kwargs["model_name_or_path"]
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        self._use_chat_template = kwargs.get("use_chat_template", True)

        self._llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self._device).eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side='left',
        )
        # Ensure pad token exists
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

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
            documents: List of document texts to process
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
            raw_prompts = [
                prompt_template.format(document=doc, question=question)
                for doc, question in zip(documents, questions)
            ]
        elif prompt_template and prompt_template.strip():
            # Paraphrase mode: format with document only
            raw_prompts = [prompt_template.format(document=doc) for doc in documents]
        else:
            # Documents are already formatted prompts
            raw_prompts = documents

        # Apply chat template if enabled (for Qwen-Instruct and similar models)
        if self._use_chat_template:
            prompts = [
                self._tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False)
                for p in raw_prompts
            ]
        else:
            prompts = raw_prompts

        # Tokenize batch with padding
        encoded = self._tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=model_args.get('max_length', 2048),
        )
        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        # Batch generation
        with torch.inference_mode():
            generated_output = self._llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self._tokenizer.pad_token_id,
                **model_args
            )

        # Decode outputs, skipping input tokens
        results = []
        for i, gen in enumerate(generated_output):
            start_idx = input_ids[i].shape[0]
            # Decode full output and extract assistant response
            full_decoded = self._tokenizer.decode(gen, skip_special_tokens=True)
            input_decoded = self._tokenizer.decode(input_ids[i], skip_special_tokens=True)

            # Remove input from output to get just the assistant response
            if self._use_chat_template:
                # For chat models, find the assistant response after </s> or similar markers
                # The output format is typically: [INST] ... [/INST] assistant response
                if input_decoded in full_decoded:
                    assistant_response = full_decoded[len(input_decoded):].strip()
                    # Clean up any remaining chat template artifacts
                    assistant_response = assistant_response.replace("system\n", "").replace("user\n", "").replace("assistant\n", "").strip()
                else:
                    assistant_response = full_decoded.strip()
            else:
                assistant_response = self._tokenizer.batch_decode(
                    gen[start_idx:],
                    skip_special_tokens=True
                )[0].strip()

            results.append(assistant_response)

        return results
