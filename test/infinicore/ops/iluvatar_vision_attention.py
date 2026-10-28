"""Qwen3.5 vision aligned-head attention, including strides and changed inputs."""

import ctypes
import os
import numpy as np
import torch
import torch.nn.functional as F
import infinicore as ic


def upload(t):
    return ic.from_torch(t.contiguous()).to(ic.device("cuda", 0))


def read(t):
    c = t.to(ic.device("cpu"))
    a = np.frombuffer(
        ctypes.string_at(c.data_ptr(), t.numel() * 2), dtype=np.uint16
    ).copy()
    return torch.from_numpy(a).view(torch.bfloat16).reshape(tuple(t.shape))


torch.set_num_threads(4)
torch.manual_seed(150)
ic.set_device(ic.device("cuda", 0))
for batch, seq, strided in [
    (1, 17, False),
    (2, 64, True),
    (1, 1216, False),
    (16, 1216, False),
    (22, 1216, False),
]:
    for repeat in range(2):
        ts = [torch.randn(batch, seq, 16, 72).bfloat16() for _ in range(3)]
        ins = [upload(t) for t in ts]
        if strided:
            ins = [
                upload(torch.stack((t, torch.zeros_like(t)), dim=2))
                .narrow(2, 0, 1)
                .view((batch, seq, 16, 72))
                for t in ts
            ]
        os.environ["INFINICORE_DISABLE_VISION_HEAD_PADDING"] = "1"
        old = read(ic.mha(*ins, alibi_slopes=None, scale=72**-0.5, is_causal=False))
        os.environ.pop("INFINICORE_DISABLE_VISION_HEAD_PADDING")
        got = read(ic.mha(*ins, alibi_slopes=None, scale=72**-0.5, is_causal=False))
        q, k, v = [F.pad(t, (0, 56)) for t in ts]
        ref = F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float(),
            v.transpose(1, 2).float(),
            scale=72**-0.5,
        ).transpose(1, 2)[..., :72]
        nrms = (
            lambda a, b: (
                (a.float() - b.float()).square().mean() / b.float().square().mean()
            )
            .sqrt()
            .item()
        )
        error = nrms(got, ref)
        delta = nrms(got, old)
        assert error < 0.01 and delta < 0.012, (batch, seq, error, delta)
        print(
            "PASS",
            batch,
            seq,
            strided,
            repeat,
            "reference_nrms",
            error,
            "old_nrms",
            delta,
            flush=True,
        )
print("ALL_VISION_ATTENTION_CHECKS_PASSED", flush=True)
