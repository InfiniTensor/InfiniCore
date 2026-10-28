"""Wide batched greedy sampling: values, first-index ties, strides, and replayed inputs."""

import ctypes
import json
import time

import infinicore as ic
import numpy as np
import torch
from infinicore.nn.functional import random_sample


def upload(tensor):
    return ic.from_torch(tensor.contiguous()).to(ic.device("cuda", 0))


def read(tensor, dtype):
    cpu = tensor.to(ic.device("cpu"))
    size = torch.empty((), dtype=dtype).element_size()
    data = ctypes.string_at(cpu.data_ptr(), tensor.numel() * size)
    np_dtype = {
        torch.bfloat16: np.uint16,
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.int32: np.int32,
    }[dtype]
    out = torch.from_numpy(np.frombuffer(data, dtype=np_dtype).copy())
    if dtype == torch.bfloat16:
        out = out.view(torch.bfloat16)
    return out.reshape(tuple(tensor.shape))


def check(batch, vocabulary, dtype, padded):
    width = vocabulary + (17 if padded else 0)
    data = torch.randn(batch, width).to(dtype)
    logits = upload(data).narrow(1, 0, vocabulary)
    for repeat in range(3):
        data = torch.randn_like(data)
        if vocabulary > 1:
            data[0, :vocabulary] = -float("inf")
            data[0, 0] = 5
            data[0, vocabulary - 1] = 5
        if vocabulary > 4096 and batch > 1:
            data[1, :vocabulary] = -float("inf")
            data[1, 4095] = float("inf")
            data[1, 4096] = float("inf")
        if batch > 2:
            data[2, :vocabulary] = -float("inf")
        # Refresh the existing strided input, as repeated engine steps do.
        logits.copy_(upload(data[:, :vocabulary]))
        values, indices = ic.topk(logits, 1, 1, largest=True, sorted=False)
        actual = read(indices, torch.int32).squeeze(1).long()
        reference = data[:, :vocabulary].float().argmax(1)
        assert torch.equal(actual, reference), (
            batch,
            vocabulary,
            dtype,
            padded,
            actual,
            reference,
        )
        expected = data[:, :vocabulary].gather(1, reference[:, None])
        assert torch.equal(read(values, dtype), expected)
        if dtype == torch.bfloat16:
            original = [
                read(
                    random_sample(
                        logits.narrow(0, i, 1).view((vocabulary,)), 0.5, 1.0, 1, 0.0
                    ),
                    torch.int32,
                ).item()
                for i in range(batch)
            ]
            # The scalar CUB sampler uses sentinel index 1 for an all -inf
            # row. Both indices tie there; the new top-k path returns index 0.
            for i, index in enumerate(original):
                if not torch.isneginf(data[i, :vocabulary]).all():
                    assert index == actual[i].item(), (i, index, actual[i].item())
    print("PASS", batch, vocabulary, str(dtype), padded, flush=True)


def benchmark():
    for batch in [1, 16, 32]:
        data = upload(torch.randn(batch, 151936).bfloat16())

        def old():
            return [
                random_sample(data.narrow(0, i, 1).view((151936,)), 0.5, 1.0, 1, 0.0)
                for i in range(batch)
            ]

        def new():
            return ic.topk(data, 1, 1, largest=True, sorted=False)

        result = {}
        for name, fn in [("per_request", old), ("batched", new)]:
            for _ in range(3):
                fn()
            ic.sync_device()
            start = time.perf_counter()
            for _ in range(100):
                fn()
            ic.sync_device()
            result[name] = (time.perf_counter() - start) * 1000 / 100
        print(
            json.dumps(dict(batch=batch, eager_host_and_device_ms=result)), flush=True
        )


if __name__ == "__main__":
    torch.set_num_threads(4)
    ic.set_device(ic.device("cuda", 0))
    torch.manual_seed(27)
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for batch, vocabulary, padded in [
            (1, 1, False),
            (3, 17, False),
            (16, 4095, False),
            (16, 4096, False),
            (32, 4097, False),
            (32, 151936, False),
            (32, 151936, True),
            (32, 248320, False),
            (32, 248320, True),
        ]:
            check(batch, vocabulary, dtype, padded)
    benchmark()
    print("ALL_BATCHED_GREEDY_CHECKS_PASSED", flush=True)
