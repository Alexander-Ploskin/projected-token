import torch
from transformers import AutoTokenizer

from evaluation.models import Model
from evaluation.models.xrag.xmistral import XMistralForCausalLM


class RAGModel(Model):
    DEFAULT_PROMPT_TEMPLATE = "Background: {document}\n Provide a paraphrase of the background sentences (background document context). Provide only the paraphrased text, no other text or formatting. Keep the meaning and all facts intact."
    default_prompt_template = DEFAULT_PROMPT_TEMPLATE

    def __init__(self, llm_name_or_path: str, device: str) -> None:
        self._device = torch.device(device)
        llm_name_or_path = llm_name_or_path
        self._llm = XMistralForCausalLM.from_pretrained(llm_name_or_path,torch_dtype = torch.bfloat16,low_cpu_mem_usage = True,).to(self._device).eval()
        self._llm_tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path,add_eos_token=False,use_fast=False,padding_side='left')

    def __call__(self, document: str, prompt_template: str, model_args: dict = {}) -> str:
        prompt = prompt_template.format_map(dict(document=document))

        input_ids = self._llm_tokenizer(prompt,return_tensors='pt').input_ids.to(self._device)
        generated_output = self._llm.generate(
                input_ids = input_ids,
                pad_token_id=self._llm_tokenizer.pad_token_id,
                **model_args
            )
        return self._llm_tokenizer.batch_decode(generated_output[:,input_ids.shape[1]:],skip_special_tokens=True)[0]
