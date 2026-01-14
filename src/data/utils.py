from __future__ import annotations
from datasets import load_dataset


def load_json_dataset(train_file: str, dev_file: str | None):
    data_files = {"train": train_file}
    if dev_file:
        data_files["dev"] = dev_file
    return load_dataset("json", data_files=data_files)
