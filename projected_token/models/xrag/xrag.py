from transformers import AutoTokenizer
import torch
from typing import List, Optional, Dict

from projected_token.models.xrag.xmistral import XMistralForCausalLM
from projected_token.models.xrag.sfr import SFR
from projected_token.models import Model


class XRAGModel(Model):
    QA_PROMPT_TEMPLATE = (
"""
Answer the Question concisely and completely, with no filler or explanation.
Document: {document}
Question: {question}
Answer:
"""
)
    PARAPHRASE_PROMPT_TEMPLATE = "Background: {document}\n Provide a paraphrase of the background sentences (background document context). Provide only the paraphrased text, no other text or formatting. Keep the meaning and all facts intact."
    default_prompt_template = QA_PROMPT_TEMPLATE

    def __init__(self, **kwargs) -> None:
        self._llm_device = torch.device(kwargs["llm_device"])
        llm_name_or_path = kwargs["llm_name_or_path"]

        llm = XMistralForCausalLM.from_pretrained(
            llm_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self._llm_device).eval()

        self._llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_name_or_path,
            add_eos_token=False,
            use_fast=False,
            padding_side="left",
        )
        self._llm_output_tokenizer = AutoTokenizer.from_pretrained(
            kwargs["tokenizer_name_or_path"],
            add_eos_token=False,
            use_fast=False,
            padding_side="left",
        )

        llm.set_xrag_token_id(self._llm_tokenizer.convert_tokens_to_ids(kwargs["xrag_token"]))
        self._xrag_token = kwargs["xrag_token"]
        self._llm = llm

        self._retriever_device = torch.device(kwargs["retriever_device"])
        retriever_name_or_path = kwargs["retriever_name_or_path"]
        self._retriever = SFR.from_pretrained(
            retriever_name_or_path,
            torch_dtype=torch.bfloat16,
        ).eval().to(self._retriever_device)
        self._retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name_or_path)

        # Cache prompt tokenization (huge win since prompt is constant for all docs)
        self._cached_prompt_template = None
        self._cached_prompt_ids = None
        self._cached_prompt_attn = None

    def _ensure_cached_prompt(self, prompt_template: str) -> None:
        if prompt_template == self._cached_prompt_template:
            return

        prompt = prompt_template.format_map({"document": self._xrag_token})
        enc = self._llm_tokenizer(prompt, return_tensors="pt")

        self._cached_prompt_ids = enc["input_ids"].to(self._llm_device)          # [1, L]
        self._cached_prompt_attn = enc.get("attention_mask", None)
        if self._cached_prompt_attn is not None:
            self._cached_prompt_attn = self._cached_prompt_attn.to(self._llm_device)  # [1, L]

        self._cached_prompt_template = prompt_template

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
            questions: Optional list of questions (for QA tasks). If provided, uses QA prompt template.

        Returns:
            List of generated answers
        """
        if model_args is None:
            model_args = {}
        if prompt_template is None:
            prompt_template = self.default_prompt_template

        # If questions provided, use QA prompt template
        if questions is not None:
            # Cache QA prompt template
            qa_template = prompt_template if prompt_template else self.QA_PROMPT_TEMPLATE
            self._ensure_cached_prompt(qa_template)
        else:
            self._ensure_cached_prompt(prompt_template)

        # 1) Batched retriever embeddings
        retr_enc = self._retriever_tokenizer(
            documents,
            max_length=180,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        retr_enc = {k: v.to(self._retriever_device) for k, v in retr_enc.items()}

        with torch.inference_mode():  # typically faster than no_grad for pure inference [web:9][web:15]
            doc_emb = self._retriever.get_doc_embedding(
                input_ids=retr_enc["input_ids"],
                attention_mask=retr_enc["attention_mask"],
            )

        # 2) Batched LLM generation
        bsz = len(documents)
        input_ids = self._cached_prompt_ids.expand(bsz, -1).contiguous()  # [B, L]

        # Important when padding batched decoder-only inputs: pass attention_mask [web:2]
        attn = None
        if self._cached_prompt_attn is not None:
            attn = self._cached_prompt_attn.expand(bsz, -1).contiguous()

        retrieval_embeds = doc_emb.to(self._llm_device)
        # If your XMistral expects [B, 1, D] instead of [B, D], uncomment:
        # retrieval_embeds = retrieval_embeds.unsqueeze(1)

        with torch.inference_mode():  # typically faster than no_grad for pure inference [web:9][web:15]
            generated = self._llm.generate(
                input_ids=input_ids,
                attention_mask=attn,
                pad_token_id=self._llm_tokenizer.pad_token_id,
                retrieval_embeds=retrieval_embeds,
                **model_args,
            )

        return self._llm_output_tokenizer.batch_decode(generated, skip_special_tokens=True)

    def __call__(self, document: str, prompt_template: str = None, model_args: Optional[Dict] = None) -> str:
        return self.generate_batch([document], prompt_template=prompt_template, model_args=model_args)[0]
