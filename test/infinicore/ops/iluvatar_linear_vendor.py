"""Exact-layout/current-stream and graph-replay regression for BF16 linear."""

import os, ctypes
import numpy as np
import torch
import infinicore as ic
from infinicore.nn.functional import linear

ic.set_device(ic.device("cuda", 0))
torch.manual_seed(7)


def read(t):
    cpu = t.to(ic.device("cpu"))
    buf = ctypes.string_at(cpu.data_ptr(), t.numel() * 2)
    return (
        torch.from_numpy(np.frombuffer(buf, dtype=np.uint16).copy())
        .view(torch.bfloat16)
        .reshape(tuple(t.shape))
        .float()
    )


passed = 0
for m, n, k in [
    (1, 4096, 4096),
    (3, 6144, 4096),
    (4, 4096, 12288),
    (16, 4096, 4096),
    (32, 4096, 4096),
    (32, 24576, 4096),
    (32, 6144, 4096),
    (16, 4096, 12288),
    (32, 4096, 12288),
    (16, 10240, 4096),
    (32, 10240, 4096),
    (4, 4096, 4096),
    (8, 4096, 12288),
    (1, 151936, 4096),
    (32, 151936, 4096),
    (1, 12352, 4096),
    (16, 12352, 4096),
    (32, 12352, 4096),
    (357, 12352, 4096),
]:
    x = (torch.randn(m, k) * 0.02).bfloat16()
    w = (torch.randn(n, k) * 0.02).bfloat16()
    a = ic.from_torch(x).to(ic.device("cuda", 0))
    b = ic.from_torch(w).to(ic.device("cuda", 0))
    y = linear(a, b)
    ic.sync_stream()
    ref = torch.nn.functional.linear(x.float(), w.float())
    actual = read(y)
    torch.testing.assert_close(actual, ref, atol=0.001, rtol=0.03)
    ic.start_graph_recording()
    z = linear(a, b)
    graph = ic.stop_graph_recording()
    for i in range(3):
        nx = (torch.randn(m, k) * 0.02).bfloat16()
        a.copy_(ic.from_torch(nx))
        graph.run()
        ic.sync_stream()
        actual = read(z)
        ref = torch.nn.functional.linear(nx.float(), w.float())
        torch.testing.assert_close(actual, ref, atol=0.001, rtol=0.03)
        passed += 1
    print("PASS", m, n, k, flush=True)
print("graph checks", passed)
