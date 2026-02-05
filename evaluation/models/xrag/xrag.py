from transformers import AutoTokenizer
import torch

from evaluation.models.xrag.xmistral import XMistralForCausalLM
from evaluation.models.xrag.sfr import SFR
from evaluation.models import Model


class XRAGModel(Model):
    DEFAULT_PROMPT_TEMPLATE = "Background: {document} Could you give me a different version of the background sentences above?"
    default_prompt_template = DEFAULT_PROMPT_TEMPLATE

    def __init__(self, **kwargs) -> None:
        self._llm_device = torch.device(kwargs["llm_device"])
        llm_name_or_path = kwargs["llm_name_or_path"]
        llm = XMistralForCausalLM.from_pretrained(llm_name_or_path,torch_dtype = torch.bfloat16,low_cpu_mem_usage = True,).to(self._llm_device).eval()
        self._llm_tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path,add_eos_token=False,use_fast=False,padding_side='left')
        llm.set_xrag_token_id(self._llm_tokenizer.convert_tokens_to_ids(kwargs["xrag_token"]))
        self._xrag_token = kwargs["xrag_token"]
        self._llm = llm

        self._retriever_device = torch.device(kwargs["retriever_device"])
        retriever_name_or_path = kwargs["retriever_name_or_path"]
        self._retriever = SFR.from_pretrained(retriever_name_or_path,torch_dtype = torch.bfloat16).eval().to(self._retriever_device)
        self._retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name_or_path)

    def __call__(self, document: str, prompt_template: str, model_args: dict = {}) -> str:
        retriever_input = self._retriever_tokenizer([document], max_length=180, padding=True, truncation=True, return_tensors='pt').to(self._retriever_device)
        with torch.no_grad():
            document_embed = self._retriever.get_doc_embedding(input_ids=retriever_input.input_ids,attention_mask=retriever_input.attention_mask)[0]
        
        prompt = prompt_template.format_map(dict(document=self._xrag_token))
        input_ids = self._llm_tokenizer(prompt,return_tensors='pt').input_ids.to(self._llm_device)
        generated_output = self._llm.generate(
                input_ids = input_ids,
                pad_token_id=self._llm_tokenizer.pad_token_id,
                retrieval_embeds = document_embed.unsqueeze(0).to(self._llm_device),
                **model_args
            )
        return self._llm_tokenizer.batch_decode(generated_output,skip_special_tokens=True)[0]
