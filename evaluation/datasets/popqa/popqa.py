from typing import Iterator, Any
import json
import pyarrow.parquet as pq

from evaluation.datasets import Dataset


class PopqaDataset(Dataset):
    def __init__(self, data: list[dict[str, Any]] = None) -> None:
        self._data = data or []

    @classmethod
    def load(cls, path: str) -> "PopqaDataset":
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as fp:
                data = [json.loads(line.strip()) for line in fp]
        elif path.endswith(".parquet"):
            # Read directly via pyarrow to avoid pandas issues with nested columns
            table = pq.read_table(path)
            # Convert to list of dicts using pyarrow's to_pydict
            pydict = table.to_pydict()
            num_rows = table.num_rows
            data = [{col: row[i] for col, row in pydict.items()} for i in range(num_rows)]
        else:
            raise ValueError(f"Unsupported format: {path}")

        return cls(data)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._data)
