import math
import random
import hashlib


def _sample_span_len(min_len: int, max_len: int, strategy: str, rng: random.Random) -> int:
    if min_len <= 0 or max_len <= 0 or min_len > max_len:
        raise ValueError(f"bad span lens: min_len={min_len} max_len={max_len}")
    if strategy == "uniform":
        return rng.randint(min_len, max_len)
    if strategy == "log_uniform":
        a, b = math.log(min_len), math.log(max_len)
        return int(round(math.exp(rng.uniform(a, b))))
    raise ValueError(f"unknown strategy={strategy}")

def crop_text_by_retriever_tokens(
    text: str,
    retriever_tokenizer,
    *,
    min_len: int,
    max_len: int,
    strategy: str = "log_uniform",
    rng: random.Random,
) -> str:
    # tokenize without adding special tokens so "length" really means content tokens
    ids = retriever_tokenizer(text, add_special_tokens=False)["input_ids"]
    n = len(ids)
    if n == 0:
        return ""
    if n <= min_len:
        return text

    L = min(_sample_span_len(min_len, max_len, strategy, rng), n)
    start = rng.randint(0, n - L)
    span_ids = ids[start : start + L]

    # decode the cropped token span back to text
    return retriever_tokenizer.decode(span_ids, skip_special_tokens=True)


def get_random(global_seed: int, example_id: str) -> random.Random:
    h = hashlib.blake2b(f"{global_seed}::{example_id}".encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(h, "little"))
