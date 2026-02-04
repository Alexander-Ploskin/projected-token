import argparse
from dataclasses import dataclass
import json
import random
from typing import Dict, Iterable
from tqdm import tqdm

import torch

from src.encoders.hf_encoder import HFMeanPoolRetriever
from src.models.factory import build_tokenizer, build_xrag_model


PROMPT_TEMPLATES = [
    "Background: {xrag_token} means the same as",
    "Background: {xrag_token} Can you put the above sentences in your own terms?",
    "Background: {xrag_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {xrag_token}\n(2)",
    "Background: {xrag_token} is a paraphrase of what?",
    "Background: {xrag_token} Could you give me a different version of the background sentences above?",
]


@dataclass(frozen=True)
class Config:
    model_name_or_path: str
    retriever_name_or_path: str
    xrag_token: str
    device: str
    ckpt_path: str

    bridge_hidden_dim: int = 1024
    bridge_dropout: float = 0.0
    use_flash_attn_2: bool = False

    retriever_max_length: int = 180

    do_sample: bool = False
    temperature: float = 0.7
    max_new_tokens: int = 400

    torch_dtype: torch.dtype = torch.bfloat16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)

    p.add_argument("--text-col", default="s_wiki_content")
    p.add_argument("--pop-col", default="s_pop")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")

    p.add_argument("--model-name-or-path", default="cache/qwen25")
    p.add_argument("--retriever-name-or-path", default="cache/retriever/retriever")
    p.add_argument("--xrag-token", default="[XRAG]")
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--ckpt-path", default="runs/xrag_hf_finewiki_sft/checkpoint-2500/projector.pt")

    p.add_argument("--bridge-hidden-dim", type=int, default=1024)
    p.add_argument("--bridge-dropout", type=float, default=0.0)
    p.add_argument("--use-flash-attn-2", action="store_true")

    p.add_argument("--retriever-max-length", type=int, default=180)

    p.add_argument("--do-sample", action="store_true", default=True)
    p.add_argument("--no-sample", dest="do_sample", action="store_false")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=40)

    args = p.parse_args()
    cfg = Config(
        model_name_or_path=args.model_name_or_path,
        retriever_name_or_path=args.retriever_name_or_path,
        xrag_token=args.xrag_token,
        device=args.device,
        ckpt_path=args.ckpt_path,
        bridge_hidden_dim=args.bridge_hidden_dim,
        bridge_dropout=args.bridge_dropout,
        use_flash_attn_2=args.use_flash_attn_2,
        retriever_max_length=args.retriever_max_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    return args, cfg


def load_retriever(cfg: Config, device: torch.device):
    retriever = HFMeanPoolRetriever(cfg.retriever_name_or_path, torch_dtype=cfg.torch_dtype)
    for p in retriever.parameters():
        p.requires_grad = False
    retriever.eval()
    retriever.to(device)
    return retriever, retriever.tokenizer, retriever.embed_dim


def load_xrag_model(cfg: Config, device: torch.device, xrag_token_id: int, retriever_embed_dim: int, tokenizer):
    model = build_xrag_model(
        model_name_or_path=cfg.model_name_or_path,
        xrag_token_id=xrag_token_id,
        retriever_embed_dim=retriever_embed_dim,
        bridge_hidden_dim=cfg.bridge_hidden_dim,
        bridge_dropout=cfg.bridge_dropout,
        use_flash_attn_2=cfg.use_flash_attn_2,
        tokenizer=tokenizer,
        torch_dtype=cfg.torch_dtype,
    )
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    model.projector.load_state_dict(ckpt, strict=True)
    model.projector.to(cfg.torch_dtype)

    model.freeze_llm()
    model.llm.eval()
    model.to(device)
    return model


def prepare_inputs_embeds(model, input_ids: torch.Tensor, retrieval_embeds: torch.Tensor, xrag_token_id: int):
    emb_layer = model.llm.get_input_embeddings()
    inputs_embeds = emb_layer(input_ids).to(torch.bfloat16)  # [B, T, H]

    retrieval_embeds = retrieval_embeds.view(-1, 1024)

    num_xrag_tokens = torch.sum(input_ids == xrag_token_id).item()
    num_retrieval_embeds = retrieval_embeds.shape[0]
    assert num_xrag_tokens == num_retrieval_embeds, (num_xrag_tokens, num_retrieval_embeds)

    retrieval_embeds = model.projector(retrieval_embeds.to(torch.bfloat16))
    inputs_embeds[input_ids == xrag_token_id] = retrieval_embeds
    return inputs_embeds


def rephrase_text(
    *,
    text: str,
    model,
    tokenizer,
    xrag_token_id: int,
    retriever,
    retriever_tokenizer,
    device: torch.device,
    cfg: Config,
    prompt_tmpl: str,
) -> str:
    retriever_input = retriever_tokenizer(
        text,
        max_length=cfg.retriever_max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    input_ids = retriever_input["input_ids"].to(device)
    attention_mask = retriever_input["attention_mask"].to(device)

    with torch.no_grad():
        doc_embed = retriever.encode(input_ids, attention_mask)[0]  # [1, D]

    retrieval_embeds = doc_embed.unsqueeze(0).repeat(1, 1)

    prompt = prompt_tmpl.format_map(dict(xrag_token=cfg.xrag_token))
    messages = [{"role": "user", "content": prompt}]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )  # [web:101]

    prompt_tokens = tokenizer(chat_text, return_tensors="pt", padding=False).to(device)
    prompt_input_ids = prompt_tokens["input_ids"]
    prompt_attention_mask = prompt_tokens["attention_mask"]

    inputs_embeds = prepare_inputs_embeds(model, prompt_input_ids, retrieval_embeds, xrag_token_id)
    assert inputs_embeds.shape[1] == prompt_attention_mask.shape[1]

    generated = model.llm.generate(
        attention_mask=prompt_attention_mask,
        inputs_embeds=inputs_embeds,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def iter_jsonl(path: str) -> Iterable[Dict]:
    # One JSON object per line (streaming). [web:51]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl_line(f, obj: Dict):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    args, cfg = parse_args()
    random.seed(args.seed)  # reproducible prompt randomization if desired. [web:63]

    device = torch.device(cfg.device)
    tokenizer, xrag_token_id = build_tokenizer(cfg.model_name_or_path, cfg.xrag_token)
    retriever, retriever_tokenizer, retriever_embed_dim = load_retriever(cfg, device)
    model = load_xrag_model(cfg, device, xrag_token_id, retriever_embed_dim, tokenizer=tokenizer)

    n_in, n_out, n_skip = 0, 0, 0

    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for row in tqdm(iter_jsonl(args.input_jsonl)):
            n_in += 1
            if args.limit and n_in > args.limit:
                break

            if args.text_col not in row:
                n_skip += 1
                continue

            text = row[args.text_col]
            if text is None or (isinstance(text, str) and text.strip() == ""):
                n_skip += 1
                continue

            prompt_tmpl = random.choice(PROMPT_TEMPLATES)  # randomized per row. [web:63]
            try:
                rephrased = rephrase_text(
                    text=text,
                    model=model,
                    tokenizer=tokenizer,
                    xrag_token_id=xrag_token_id,
                    retriever=retriever,
                    retriever_tokenizer=retriever_tokenizer,
                    device=device,
                    cfg=cfg,
                    prompt_tmpl=prompt_tmpl,
                )
            except Exception as e:
                print(e)
                # Keep going; store error text if you want, or just skip.
                n_skip += 1
                continue

            out_row = {
                args.pop_col: row.get(args.pop_col, None),
                args.text_col: text,
                "rephrased_text": rephrased,
            }
            write_jsonl_line(out_f, out_row)
            n_out += 1

    print(f"done: in={n_in} out={n_out} skipped={n_skip} -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
