from typing import Iterator, Any
import json
import pyarrow.parquet as pq

from evaluation.datasets import Dataset


class PopqaDataset(Dataset):
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self._data = data

    @classmethod
    def load(cls, path: str) -> "PopqaDataset":
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as fp:
                data = [json.loads(line.strip()) for line in fp]
        elif path.endswith(".parquet"):
            table = pq.read_table(path)
            df = table.to_pandas()
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported format: {path}")
        
        return cls(data)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._data)
