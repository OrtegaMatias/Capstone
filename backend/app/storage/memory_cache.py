from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class CachedDataset:
    dataset_id: str
    dataframe: pd.DataFrame
    metadata: dict[str, Any]


class DatasetMemoryCache:
    def __init__(self, max_items: int = 10):
        self.max_items = max_items
        self._cache: OrderedDict[str, CachedDataset] = OrderedDict()

    def get(self, dataset_id: str) -> CachedDataset | None:
        if dataset_id not in self._cache:
            return None
        item = self._cache.pop(dataset_id)
        self._cache[dataset_id] = item
        return item

    def set(self, item: CachedDataset) -> None:
        if item.dataset_id in self._cache:
            self._cache.pop(item.dataset_id)
        self._cache[item.dataset_id] = item
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
