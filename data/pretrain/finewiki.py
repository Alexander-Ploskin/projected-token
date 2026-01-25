import os
from datasets import load_dataset, DownloadConfig
from sklearn.model_selection import train_test_split


ds = load_dataset(
    "HuggingFaceFW/finewiki",
    name="en",
    split="train",
    cache_dir="./data/hf_datasets_cache",
    download_config=DownloadConfig(disable_tqdm=False),
)

df = ds.to_pandas()
bad = df.applymap(lambda x: x is Ellipsis).any()
print("Columns containing Ellipsis:", bad[bad].index.tolist())

train_df, dev_df = train_test_split(df, test_size=0.005, random_state=42)


train_df.to_json("finewiki_train.jsonl", orient="records", lines=True, force_ascii=False)
dev_df.to_json("finewiki_dev.jsonl", orient="records", lines=True, force_ascii=False)