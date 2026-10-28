"""Qwen BF16 prefill regression: padding/strides, exact KV writes, and graph replay."""

import ctypes
import numpy as np
import torch
import infinicore as ic


def upload(t):
    return ic.from_torch(t.contiguous()).to(ic.device("cuda", 0))


def read(t):
    cpu = t.contiguous().to(ic.device("cpu"))
    raw = ctypes.string_at(cpu.data_ptr(), t.numel() * 2)
    return (
        torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy())
        .view(torch.bfloat16)
        .reshape(tuple(t.shape))
    )


def check_cache(m, h, dk, dv, permuted=False, unaligned=False):
    bs = 64
    nb = (m + 63) // 64 + 2
    # Padding between source heads must not be copied into either cache.
    sk = dk + (1 if unaligned else 8)
    sv = dv + (1 if unaligned else 8)
    kt = torch.randn(m, h, sk).bfloat16()
    vt = torch.randn(m, h, sv).bfloat16()
    k = upload(kt).as_strided([m, h, dk], [h * sk, sk, 1])
    v = upload(vt).as_strided([m, h, dv], [h * sv, sv, 1])
    shape = [nb, bs, h, dk] if permuted else [nb, h, bs, dk]
    vshape = [nb, bs, h, max(dk, dv)] if permuted else [nb, h, bs, max(dk, dv)]
    kc, vc = upload(torch.full(shape, -3.0).bfloat16()), upload(
        torch.full(vshape, -3.0).bfloat16()
    )
    if permuted:
        kc, vc = kc.permute([0, 2, 1, 3]), vc.permute([0, 2, 1, 3])
    slots = torch.randperm(nb * bs)[:m].long()
    slots[::11] = -1
    slot = upload(slots)
    # Integer upload above keeps its dtype; read() is used only for BF16.
    ic.paged_caching(kc, vc, k, v, slot)
    ic.sync_stream()
    kr, vr = read(kc), read(vc)
    valid = slots >= 0
    kr0 = torch.full_like(kr, -3)
    vr0 = torch.full_like(vr, -3)
    kr0[slots[valid] // bs, :, slots[valid] % bs, :] = kt[valid, :, :dk]
    vr0[slots[valid] // bs, :, slots[valid] % bs, :dv] = vt[valid, :, :dv]
    assert torch.equal(kr, kr0) and torch.equal(vr, vr0)
    ic.start_graph_recording()
    ic.paged_caching(kc, vc, k, v, slot)
    graph = ic.stop_graph_recording()
    for replay in range(3):
        kt = torch.randn_like(kt)
        vt = torch.randn_like(vt)
        k.copy_(upload(kt[:, :, :dk]))
        v.copy_(upload(vt[:, :, :dv]))
        slots = torch.randperm(nb * bs)[:m].long()
        slots[::11] = -1
        slot.copy_(upload(slots))
        valid = slots >= 0
        kr0[slots[valid] // bs, :, slots[valid] % bs, :] = kt[valid, :, :dk]
        vr0[slots[valid] // bs, :, slots[valid] % bs, :dv] = vt[valid, :, :dv]
        graph.run()
        ic.sync_stream()
        assert torch.equal(read(kc), kr0) and torch.equal(read(vc), vr0)
    print("PASS cache", m, h, dk, dv, permuted, unaligned, flush=True)


def check_elementwise(m, n):
    from infinicore.nn.functional import swiglu

    up = torch.randn(m, n).bfloat16()
    gate = torch.randn(m, n).bfloat16()
    a, b = upload(up), upload(gate)
    c = swiglu(a, b)

    def verify(u, g):
        sig = (1 / (1 + torch.exp(-g.float()))).bfloat16().float()
        ref = ((g.float() * sig) * u.float()).bfloat16()
        torch.testing.assert_close(read(c), ref, atol=0, rtol=0)

    verify(up, gate)
    ic.start_graph_recording()
    c = swiglu(a, b)
    graph = ic.stop_graph_recording()
    for _ in range(3):
        up = torch.randn_like(up)
        gate = torch.randn_like(gate)
        a.copy_(upload(up))
        b.copy_(upload(gate))
        graph.run()
        ic.sync_stream()
        verify(up, gate)
    print("PASS swiglu", m, n, flush=True)
    n = 4096
    av, bv = torch.randn(m, n).bfloat16(), torch.randn(m, n).bfloat16()
    w = torch.randn(n).bfloat16()
    a, b, weight = upload(av), upload(bv), upload(w)
    y, r = ic.add_rms_norm(a, b, weight, 1e-6)

    def verify_norm():
        summed = av.float() + bv.float()
        rr = summed.bfloat16()
        ref = (
            rr.float()
            * w.float()
            * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
        ).bfloat16()
        assert torch.equal(read(r), rr)
        torch.testing.assert_close(read(y), ref, atol=0.03125, rtol=0.008)

    verify_norm()
    ic.start_graph_recording()
    y, r = ic.add_rms_norm(a, b, weight, 1e-6)
    graph = ic.stop_graph_recording()
    for _ in range(3):
        av, bv = torch.randn_like(av), torch.randn_like(bv)
        a.copy_(upload(av))
        b.copy_(upload(bv))
        graph.run()
        ic.sync_stream()
        verify_norm()
    print("PASS add_rms_norm", m, n, flush=True)


if __name__ == "__main__":
    torch.manual_seed(1234)
    ic.set_device(ic.device("cuda", 0))
    for m in [1, 16, 32, 53, 1045, 8192]:
        check_elementwise(m, 12288)
    for m, h, dk, dv in [
        (1, 8, 128, 128),
        (32, 8, 128, 128),
        (129, 8, 128, 128),
        (1045, 8, 128, 128),
        (11473, 8, 128, 128),
        (357, 4, 256, 256),
        (135, 1, 576, 512),
        (135, 4, 192, 128),
    ]:
        check_cache(m, h, dk, dv)
        check_cache(m, h, dk, dv, True)
    check_cache(129, 8, 127, 127, False, True)
    print("All prefill eager/graph checks passed", flush=True)
