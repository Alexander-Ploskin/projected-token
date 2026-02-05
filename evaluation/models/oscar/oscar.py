import torch
from transformers import AutoModel

from evaluation.models import Model


class OscarModel(Model):
    DEFAULT_PROMPT_TEMPLATE = "Provide a paraphrase of the background sentences (background document context). Provide only the paraphrased text, no other text or formatting. Keep the meaning and all facts intact."
    default_prompt_template = DEFAULT_PROMPT_TEMPLATE

    def __init__(self, **kwargs) -> None:
        device = torch.device(kwargs["device"])
        model_name_or_path = kwargs["model_name_or_path"]
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        trust_remote_code = kwargs.get("trust_remote_code", True)
        self._model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(device).eval()

    # OSCAR's generate_from_compressed_documents_and_questions does not accept HF-style
    # kwargs like do_sample/temperature; only pass whitelisted args.
    _GENERATION_WHITELIST = ("max_new_tokens",)

    def __call__(self, document: str, prompt_template: str, model_args: dict = None) -> str:
        if model_args is None:
            model_args = {}
        if not document.strip():
            return ""
        question = prompt_template
        emb = self._model.compress_documents(documents=[document], questions=[question])
        gen_kwargs = {k: v for k, v in model_args.items() if k in self._GENERATION_WHITELIST}
        out = self._model.generate_from_compressed_documents_and_questions(
            questions=[question], compressed_documents=emb, **gen_kwargs
        )
        return out[0] if out else ""
