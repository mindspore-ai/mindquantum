#!/usr/bin/env python3
"""Inspect the block coordinates observed by the mqvector_npu Ascend C kernel."""

from __future__ import annotations

import argparse
import json

import numpy as np


def parse_block_dims(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive Ascend C block dimensions."""
    try:
        block_dims = tuple(int(item) for item in value.split(",") if item)
    except ValueError as err:
        raise argparse.ArgumentTypeError("block dimensions must be integers") from err
    if not block_dims or any(block_dim <= 0 or block_dim > 64 for block_dim in block_dims):
        raise argparse.ArgumentTypeError("block dimensions must be between 1 and 64")
    return block_dims


def main() -> None:
    """Run the device-side block-coordinate probe and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-dims", type=parse_block_dims, default=(1, 2, 4, 8, 16, 32))
    args = parser.parse_args()

    from mindquantum import _mq_vector_npu

    sentinel = int(np.iinfo(np.uint32).max)
    results = []
    for block_dim in args.block_dims:
        observed = _mq_vector_npu._probe_block_dim(block_dim).tolist()
        results.append(
            {
                "requested_block_dim": block_dim,
                "reported_block_num_by_slot": [None if value == sentinel else value for value in observed],
                "covered_slots": sum(value != sentinel for value in observed),
            }
        )

    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
