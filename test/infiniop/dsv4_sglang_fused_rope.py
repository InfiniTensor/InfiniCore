import ctypes
from ctypes import c_bool, c_size_t

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

_CASES = [((128, 64),), ((1, 64),), ((64, 8),)]
_ROPE_DIM = 64


def _workspace(desc, getter, device):
    size = c_size_t(0)
    check_error(getter(desc, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


def _ref_rotary_emb(q, freqs_cis):
    q_float = q.float().reshape(*q.shape[:-1], -1, 2)
    q_complex = torch.view_as_complex(q_float)
    out = torch.view_as_real(q_complex * freqs_cis[:, None, :]).flatten(-2)
    return out.to(q.dtype)


def test_op(handle, device, case, dtype=InfiniDtype.BF16, sync=None):
    m, n_heads = case
    print(
        f"Testing DSV4 sglang_fused_rope on {InfiniDeviceNames[device]} m:{m} heads:{n_heads}"
    )
    torch.manual_seed(m + n_heads)
    q_init = torch.randn(
        m, n_heads, _ROPE_DIM, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    freqs_cis = torch.randn(m, _ROPE_DIM // 2, dtype=torch.complex64, device="cuda")
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    positions = torch.arange(m, dtype=torch.int64, device="cuda")
    ref = _ref_rotary_emb(q_init, freqs_cis)

    q = TestTensor.from_torch(q_init.clone(), dtype, device)
    freqs = TestTensor.from_torch(freqs_real, InfiniDtype.F32, device)
    pos = TestTensor.from_torch(positions, InfiniDtype.I64, device)

    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangFusedRopeDescriptor(
            handle,
            ctypes.byref(desc),
            q.descriptor,
            freqs.descriptor,
            pos.descriptor,
            c_bool(False),
        )
    )
    for tensor in [q, freqs, pos]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc,
        LIBINFINIOP.infiniopGetDsv4SglangFusedRopeWorkspaceSize,
        device,
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangFusedRope(
            desc,
            workspace.data(),
            workspace_size,
            q.data(),
            freqs.data(),
            pos.data(),
            None,
        )
    )
    assert torch.allclose(q.actual_tensor(), ref, atol=1e-2, rtol=1e-2)
    check_error(LIBINFINIOP.infiniopDestroyDsv4SglangFusedRopeDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 sglang_fused_rope Test passed!\033[0m")
