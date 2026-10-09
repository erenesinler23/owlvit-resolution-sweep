"""Block-level development/test split.

Chips are grouped into geographic blocks and whole blocks go to one split, so
development tuning cannot leak into the test set through neighbouring chips.
The cell size is in the units of the chip coordinate reference system.
"""
from __future__ import annotations

import zlib
from typing import Sequence

import numpy as np


def block_key(city: str, x: float, y: float, cell: float) -> str:
    return f"{city}:{int(np.floor(x / cell))}:{int(np.floor(y / cell))}"


def assign_split(chip_ids: Sequence[str], cities: Sequence[str],
                 xs: Sequence[float], ys: Sequence[float], cell: float,
                 dev_fraction: float, seed: int) -> list[dict]:
    """Return one record per chip with block id and split ('dev' or 'test').

    Blocks are shuffled per city with a seed derived from the city name, and
    the first dev_fraction of them become the development set.
    """
    blocks = [block_key(c, x, y, cell) for c, x, y in zip(cities, xs, ys)]
    split_of_block: dict[str, str] = {}
    for city in sorted(set(cities)):
        city_blocks = sorted({b for b, c in zip(blocks, cities) if c == city})
        rng = np.random.default_rng(seed + zlib.crc32(city.encode()))
        order = rng.permutation(len(city_blocks))
        n_dev = int(round(dev_fraction * len(city_blocks)))
        if len(city_blocks) > 1:
            n_dev = min(max(n_dev, 1), len(city_blocks) - 1)
        for rank, i in enumerate(order):
            split_of_block[city_blocks[i]] = "dev" if rank < n_dev else "test"
    return [
        {"chip_id": cid, "city": city, "block": blk, "split": split_of_block[blk]}
        for cid, city, blk in zip(chip_ids, cities, blocks)
    ]


def subsample(records: Sequence[dict], n: int, seed: int) -> list[dict]:
    """Seeded uniform subsample of records, returned sorted by chip id."""
    records = sorted(records, key=lambda r: r["chip_id"])
    if n >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(records), size=n, replace=False))
    return [records[i] for i in pick]
