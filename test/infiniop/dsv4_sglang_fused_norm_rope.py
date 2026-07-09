import ctypes
from ctypes import c_double, c_size_t

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    TestTensor,
    TestWorkspace,
    check_error,
    get_args,
    get_test_devices,
    infiniopOperatorDescriptor_t,
    test_operator,
)


def _workspace(desc, getter, device):
    size = c_size_t(0)
    check_error(getter(desc, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


_EPS = 1e-6


def _ref_norm_rope_tail(x, freqs_real, positions, eps=1e-6, weight=None):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    rope_dim = freqs_real.shape[-1]
    tail = y[..., -rope_dim:]
    tail_even = tail[..., 0::2]
    tail_odd = tail[..., 1::2]
    freq = freqs_real[positions.long()]
    while freq.ndim < tail.ndim:
        freq = freq.unsqueeze(-2)
    cos = freq[..., 0::2]
    sin = freq[..., 1::2]
    rotated = torch.empty_like(tail)
    rotated[..., 0::2] = tail_even * cos - tail_odd * sin
    rotated[..., 1::2] = tail_even * sin + tail_odd * cos
    y[..., -rope_dim:] = rotated
    return y.to(x.dtype)


_CASES = [(5,)]


def test_op(handle, device, tokens, dtype=InfiniDtype.BF16, sync=None):
    print(
        f"Testing DSV4 sglang_fused_norm_rope on {InfiniDeviceNames[device]} tokens:{tokens}"
    )
    head_dim, rope_dim = 512, 64
    kv_torch = torch.randn(
        tokens, head_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    original = kv_torch.clone()
    weight_torch = torch.randn(
        head_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    positions_torch = torch.tensor([0, 3, 7, 11, 13], dtype=torch.int64, device="cuda")[
        :tokens
    ]
    freqs_cis = torch.randn(16, rope_dim // 2, dtype=torch.complex64, device="cuda")
    freqs_torch = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    kv = TestTensor.from_torch(kv_torch, dtype, device)
    weight = TestTensor.from_torch(weight_torch, dtype, device)
    positions = TestTensor.from_torch(positions_torch, InfiniDtype.I64, device)
    freqs = TestTensor.from_torch(freqs_torch, InfiniDtype.F32, device)
    ref = _ref_norm_rope_tail(
        original, freqs_torch, positions_torch, _EPS, weight_torch
    )
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangFusedNormRopeDescriptor(
            handle,
            ctypes.byref(desc),
            kv.descriptor,
            weight.descriptor,
            positions.descriptor,
            freqs.descriptor,
            c_double(_EPS),
        )
    )
    for tensor in [kv, weight, positions, freqs]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4SglangFusedNormRopeWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangFusedNormRope(
            desc,
            workspace.data(),
            workspace_size,
            kv.data(),
            weight.data(),
            positions.data(),
            freqs.data(),
            None,
        )
    )
    assert torch.allclose(kv.actual_tensor(), ref, atol=2e-2, rtol=2e-2)
    check_error(LIBINFINIOP.infiniopDestroyDsv4SglangFusedNormRopeDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 sglang_fused_norm_rope Test passed!\033[0m")
