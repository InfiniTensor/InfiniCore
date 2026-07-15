#!/usr/bin/env python3
"""Focused cross-device event ordering test for pipeline transport."""

import itertools
import torch

import infinicore


def check_pair(source_id, destination_id, expected):
    source_device = infinicore.device("cuda", source_id)
    destination_device = infinicore.device("cuda", destination_id)
    source = infinicore.from_torch(expected).to(source_device)

    source_ready = infinicore.DeviceEvent(device=source_device)
    source_ready.record()
    source_ready.wait_on(destination_device)

    destination_ready = infinicore.DeviceEvent(device=destination_device)
    destination_ready.record()
    destination_ready.synchronize()
    assert source_ready.query()

    copied = source.to(destination_device).to(infinicore.device("cpu", 0))
    actual = torch.empty_like(expected)
    infinicore.from_torch(actual).copy_(copied)
    assert torch.equal(actual, expected)


def main():
    device_count = infinicore.get_device_count("cuda")
    if device_count < 2:
        raise RuntimeError(
            f"cross-device event test requires two GPUs, found {device_count}"
        )

    expected = torch.arange(4096, dtype=torch.float32)
    pairs = list(itertools.permutations(range(device_count), 2))
    for source_id, destination_id in pairs:
        check_pair(source_id, destination_id, expected)

    print(
        f"PASS: cross-device event wait and peer copy on {len(pairs)} directed pairs"
    )


if __name__ == "__main__":
    main()
