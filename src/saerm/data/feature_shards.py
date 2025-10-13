import os
import json
import torch
from typing import Iterator
from torch.utils.data import IterableDataset


class FeatureShardBatches(IterableDataset):
    def __init__(self, dir: str, batch_size: int, pin_memory: bool = False):
        super().__init__()
        self.dir = dir
        self.batch_size = int(batch_size)
        self.pin_memory = pin_memory
        with open(os.path.join(dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)

    @property
    def dim(self) -> int:
        return int(self.manifest["dim"])  # type: ignore[index]

    def __iter__(self) -> Iterator[torch.Tensor]:
        shards = self.manifest["shards"]  # type: ignore[index]
        for sh in shards:
            path = os.path.join(self.dir, sh["file"])  # type: ignore[index]
            t = torch.load(path, map_location="cpu")
            t = t.float()  # keep training stable, move to device per-batch
            n = t.shape[0]
            for i in range(0, n, self.batch_size):
                yield t[i : i + self.batch_size]


