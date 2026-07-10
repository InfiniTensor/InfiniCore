import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import infinicore
from framework import get_args, get_test_devices, torch_device_map, InfiniDeviceEnum, to_torch_dtype, convert_infinicore_to_torch

ACT_SILU = 0
ACT_SWIGLU = 1
CASES = [
    (2, 16, 32, 4, 2, ACT_SILU),
    (3, 32, 16, 5, 2, ACT_SWIGLU),
    (1, 2560, 1536, 64, 6, ACT_SWIGLU),
    (64, 2560, 1536, 64, 6, ACT_SWIGLU),
]
DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]
TOLS = {
    infinicore.float16: {"atol": 2e-2, "rtol": 2e-2},
    infinicore.bfloat16: {"atol": 5e-2, "rtol": 5e-2},
    infinicore.float32: {"atol": 1e-4, "rtol": 1e-4},
}


def ref(x, indices, scales, w1, w2, b1, b2, activation):
    N, hidden = x.shape
    topk = indices.shape[1]
    out = torch.zeros((N, hidden), dtype=torch.float32, device=x.device)
    for n in range(N):
        x_f = x[n].float()
        for k in range(topk):
            e = int(indices[n, k])
            h1 = w1[e].float() @ x_f
            if b1 is not None:
                h1 = h1 + b1[e].float()
            if activation == ACT_SWIGLU:
                gate, up = h1.chunk(2, dim=0)
                act = F.silu(gate) * up
            else:
                act = F.silu(h1)
            y = w2[e].float() @ act
            if b2 is not None:
                y = y + b2[e].float()
            out[n] += scales[n, k].float() * y
    return out.to(x.dtype)


def wrap(t):
    return infinicore.from_torch(t.contiguous())


def run_case(device, case, dtype):
    N, hidden, inter, experts, topk, activation = case
    torch_device = torch_device_map[device]
    torch_dtype = to_torch_dtype(dtype)
    print(f"Testing InfiniCore fused_moe N={N} hidden={hidden} inter={inter} experts={experts} topk={topk} activation={activation} dtype={dtype}")
    x = (torch.rand((N, hidden), dtype=torch_dtype, device=torch_device) * 2 - 1).contiguous()
    w1_cols = inter * 2 if activation == ACT_SWIGLU else inter
    w1 = (torch.rand((experts, w1_cols, hidden), dtype=torch_dtype, device=torch_device) * 2 - 1).contiguous()
    w2 = (torch.rand((experts, hidden, inter), dtype=torch_dtype, device=torch_device) * 2 - 1).contiguous()
    b1 = (torch.rand((experts, w1_cols), dtype=torch_dtype, device=torch_device) * 0.1).contiguous()
    b2 = (torch.rand((experts, hidden), dtype=torch_dtype, device=torch_device) * 0.1).contiguous()
    logits = torch.rand((N, experts), dtype=torch.float32, device=torch_device)
    scales, indices64 = torch.topk(F.softmax(logits, dim=-1), topk, dim=-1)
    scales = (scales / scales.sum(dim=-1, keepdim=True)).contiguous()
    indices = indices64.to(torch.int32).contiguous()
    ans = ref(x, indices, scales, w1, w2, b1, b2, activation)

    out = infinicore.nn.functional.fused_moe(
        wrap(x), wrap(indices), wrap(scales), wrap(w1), wrap(w2), b1=wrap(b1), b2=wrap(b2), activation=activation
    )
    infinicore.sync_device()
    actual = convert_infinicore_to_torch(out)
    assert torch.allclose(actual, ans, **TOLS[dtype])

    out_inplace = infinicore.empty((N, hidden), dtype=dtype, device=infinicore.device(torch_device, 0))
    returned = infinicore.nn.functional.fused_moe(
        wrap(x), wrap(indices), wrap(scales), wrap(w1), wrap(w2), b1=wrap(b1), b2=wrap(b2), activation=activation, out=out_inplace
    )
    infinicore.sync_device()
    assert returned is out_inplace
    actual_inplace = convert_infinicore_to_torch(out_inplace)
    assert torch.allclose(actual_inplace, ans, **TOLS[dtype])


def main():
    args = get_args()
    for device in get_test_devices(args):
        if device != InfiniDeviceEnum.NVIDIA:
            continue
        infinicore.set_device(infinicore.device(torch_device_map[device], 0))
        for case in CASES:
            for dtype in DTYPES:
                run_case(device, case, dtype)
    print("\033[92mTest passed!\033[0m")


if __name__ == "__main__":
    main()
