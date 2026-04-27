import torch
from transformers import AutoModel
from typing import List, Optional, Dict

from projected_token.models import Model


class OscarModel(Model):
    QA_DEFAULT_PROMPT_TEMPLATE = (
"""Answer the Question concisely and completely, with no filler or explanation.
Question: {question}
Answer:
"""
)
    
    PARAPHRASE_DEFAULT_PROMPT_TEMPLATE = "Provide a paraphrase of the background sentences (background document context). Provide only the paraphrased text, no other text or formatting. Keep the meaning and all facts intact."
    
    default_prompt_template = QA_DEFAULT_PROMPT_TEMPLATE

    def __init__(self, **kwargs) -> None:
        device = kwargs.get("device", "cuda:0")
        model_name_or_path = kwargs["model_name_or_path"]
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        trust_remote_code = kwargs.get("trust_remote_code", True)

        # Force all components (including compressor) to use specified device
        # Use device_map="cuda:0" explicitly to avoid auto-assignment to other GPUs
        device_map = device if device != "cpu" else "cpu"

        self._model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        ).eval()

    # OSCAR's generate_from_compressed_documents_and_questions does not accept HF-style
    # kwargs like do_sample/temperature; only pass whitelisted args.
    _GENERATION_WHITELIST = ("max_new_tokens",)

    def __call__(self, document: str, prompt_template: str, model_args: dict = None) -> str:
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
            prompt_template: Template for questions (used if questions not provided)
            model_args: Generation arguments (only max_new_tokens is used)
            questions: Optional list of questions. If provided, used directly for each document.

        Returns:
            List of generated answers
        """
        if model_args is None:
            model_args = {}

        # Filter empty documents
        valid_indices = [i for i, doc in enumerate(documents) if doc.strip()]
        valid_docs = [documents[i] for i in valid_indices]

        if not valid_docs:
            return [""] * len(documents)

        # Use provided questions or fall back to prompt_template
        if questions is not None:
            qa_questions = [questions[i] for i in valid_indices]
        else:
            qa_questions = [prompt_template] * len(valid_docs) if prompt_template else [""] * len(valid_docs)

        # Batch compress documents
        gen_kwargs = {k: v for k, v in model_args.items() if k in self._GENERATION_WHITELIST}

        with torch.inference_mode():
            emb = self._model.compress_documents(documents=valid_docs, questions=qa_questions)
            out = self._model.generate_from_compressed_documents_and_questions(
                questions=qa_questions, compressed_documents=emb, **gen_kwargs
            )

        # Map results back to original order
        results = [""] * len(documents)
        for i, idx in enumerate(valid_indices):
            results[idx] = out[i] if i < len(out) else ""

        return results
