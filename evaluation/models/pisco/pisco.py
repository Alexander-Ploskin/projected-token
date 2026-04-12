import os
import sys
import torch
from typing import Dict, List, Optional

from transformers import AutoConfig

from evaluation.models import Model


class PiscoModel(Model):
    """PISCO model for compressed context retrieval.

    PISCO (https://github.com/naver-ai/pisco) compresses documents into
    fixed-size latent representations and generates answers from the
    compressed representation using COCOM architecture.
    """

    QA_DEFAULT_PROMPT_TEMPLATE = (
        "Answer the Question concisely and completely, with no filler or explanation.\n"
        "Question: {question}\n"
        "Answer:\n"
    )

    PARAPHRASE_DEFAULT_PROMPT_TEMPLATE = (
        "Provide a paraphrase of the background sentences (background document context). "
        "Provide only the paraphrased text, no other text or formatting. "
        "Keep the meaning and all facts intact."
    )

    default_prompt_template = QA_DEFAULT_PROMPT_TEMPLATE

    def __init__(self, **kwargs) -> None:
        device = torch.device(kwargs["device"])
        model_name_or_path = kwargs["model_name_or_path"]
        trust_remote_code = kwargs.get("trust_remote_code", True)

        from huggingface_hub import hf_hub_download

        modelling_path = hf_hub_download(
            repo_id=model_name_or_path,
            filename="modelling_pisco.py",
        )
        sys.path.insert(0, os.path.dirname(modelling_path))

        self._config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        from modelling_pisco import COCOM, add_memory_tokens_to_inputs

        self._model = COCOM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        ).to(device).eval()
        self._device = device
        self._tokenizer = self._model.decoder_tokenizer
        self._add_mem_tokens = add_memory_tokens_to_inputs

        self._doc_max_length = self._config.doc_max_length
        self._compr_rate = self._config.compr_rate
        self._num_mem_tokens = self._doc_max_length // self._compr_rate

    def __call__(
        self, document: str, prompt_template: str, model_args: Optional[Dict] = None
    ) -> str:
        return self.generate_batch([document], prompt_template, model_args)[0]

    def generate_batch(
        self,
        documents: List[str],
        prompt_template: Optional[str] = None,
        model_args: Optional[Dict] = None,
        questions: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate answers for a batch of documents using PISCO's compressed context.

        Args:
            documents: List of document texts to process.
            prompt_template: Template for questions (used if questions not provided).
            model_args: Generation arguments (max_new_tokens, etc.).
            questions: Optional list of questions. If provided, used directly for each document.

        Returns:
            List of generated answers.
        """
        if model_args is None:
            model_args = {}
        max_new_tokens = model_args.get("max_new_tokens", 512)

        valid_indices = [i for i, doc in enumerate(documents) if doc.strip()]
        valid_docs = [documents[i] for i in valid_indices]

        if not valid_docs:
            return [""] * len(documents)

        if questions is not None:
            qa_questions = [questions[i] for i in valid_indices]
        else:
            qa_questions = (
                [prompt_template] * len(valid_docs) if prompt_template else [""] * len(valid_docs)
            )

        results = []
        with torch.inference_mode():
            for doc, question in zip(valid_docs, qa_questions):
                enc_inputs = self._model.prepare_encoder_inputs(
                    texts=[doc],
                    max_length=self._doc_max_length,
                    q_texts=[question] if question else None,
                )

                dec_text = f"<AE>{question}"
                dec_enc = self._tokenizer(
                    dec_text, return_tensors="pt", add_special_tokens=False
                )

                dec_enc["input_ids"], dec_enc["attention_mask"] = self._add_mem_tokens(
                    dec_enc["input_ids"],
                    dec_enc["attention_mask"],
                    self._num_mem_tokens,
                    tokenizer=self._tokenizer,
                )

                model_input = {
                    "enc_input_ids": enc_inputs["input_ids"].to(self._device),
                    "enc_attention_mask": enc_inputs["attention_mask"].to(self._device),
                    "dec_input_ids": dec_enc["input_ids"].to(self._device),
                    "dec_attention_mask": dec_enc["attention_mask"].to(self._device),
                }

                output = self._model.generate(model_input, max_new_tokens=max_new_tokens)
                results.append(output[0] if isinstance(output, list) else output)

        final_results = [""] * len(documents)
        for i, idx in enumerate(valid_indices):
            final_results[idx] = results[i] if i < len(results) else ""

        return final_results
