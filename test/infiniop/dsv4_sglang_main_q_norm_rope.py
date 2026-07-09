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


_CASES = [((3, 4, 512),)]


def test_op(handle, device, shape, dtype=InfiniDtype.BF16, sync=None):
    print(
        f"Testing DSV4 sglang_main_q_norm_rope on {InfiniDeviceNames[device]} shape:{shape}"
    )
    batch, _, _ = shape
    rope_dim = 64
    q_torch = torch.randn(shape, dtype=torch.bfloat16, device="cuda").contiguous()
    positions_torch = torch.tensor([1, 5, 9], dtype=torch.int32, device="cuda")[:batch]
    freqs_cis = torch.randn(12, rope_dim // 2, dtype=torch.complex64, device="cuda")
    freqs_torch = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    output = TestTensor(shape, None, dtype, device, mode="zeros")
    q = TestTensor.from_torch(q_torch, dtype, device)
    freqs = TestTensor.from_torch(freqs_torch, InfiniDtype.F32, device)
    positions = TestTensor.from_torch(positions_torch, InfiniDtype.I32, device)
    ref = _ref_norm_rope_tail(q_torch, freqs_torch, positions_torch, _EPS, None)
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangMainQNormRopeDescriptor(
            handle,
            ctypes.byref(desc),
            output.descriptor,
            q.descriptor,
            freqs.descriptor,
            positions.descriptor,
            c_double(_EPS),
        )
    )
    for tensor in [output, q, freqs, positions]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4SglangMainQNormRopeWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangMainQNormRope(
            desc,
            workspace.data(),
            workspace_size,
            output.data(),
            q.data(),
            freqs.data(),
            positions.data(),
            None,
        )
    )
    assert torch.allclose(output.actual_tensor(), ref, atol=2e-2, rtol=2e-2)
    check_error(LIBINFINIOP.infiniopDestroyDsv4SglangMainQNormRopeDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 sglang_main_q_norm_rope Test passed!\033[0m")
