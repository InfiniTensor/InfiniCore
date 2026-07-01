import ctypes
from ctypes import c_uint64

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    test_operator,
)

_BASE_TEST_CASES = [
    # num_tokens, num_q_heads, num_kv_heads, head_size, rotary_dim, sections, interleaved
    (5, 2, 1, 128, 128, (16, 24, 24), False),
    (7, 4, 2, 128, 128, (16, 24, 24), False),
    (3, 3, 1, 128, 96, (8, 16, 24), False),
    (6, 2, 2, 128, 96, (8, 16, 24), True),
]
_POSITION_DTYPES = [InfiniDtype.I32, InfiniDtype.I64]
_TEST_CASES = [
    (*case, position_dtype)
    for case in _BASE_TEST_CASES
    for position_dtype in _POSITION_DTYPES
]
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 2e-2, "rtol": 2e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}
DEBUG = False


def axis_for_dim(dim, sections, interleaved):
    t, h, w = sections
    if interleaved:
        h_mask = dim % 3 == 1 and dim < h * 3
        w_mask = dim % 3 == 2 and dim < w * 3
        return 1 if h_mask else (2 if w_mask else 0)
    if dim < t:
        return 0
    if dim < t + h:
        return 1
    return 2


def torch_mrope_one(
    x, cos, sin, positions, head_size, rotary_dim, sections, interleaved
):
    num_tokens = x.shape[0]
    num_heads = x.shape[1] // head_size
    x = x.reshape(num_tokens, num_heads, head_size)
    out = x.clone()
    half = rotary_dim // 2
    cos_row = torch.empty((num_tokens, half), dtype=cos.dtype, device=cos.device)
    sin_row = torch.empty((num_tokens, half), dtype=sin.dtype, device=sin.device)
    has_axes = positions.ndim == 2
    for i in range(half):
        axis = axis_for_dim(i, sections, interleaved)
        pos = positions[axis] if has_axes else positions
        cos_row[:, i] = cos[pos, i]
        sin_row[:, i] = sin[pos, i]
    x0 = x[:, :, :half].float()
    x1 = x[:, :, half:rotary_dim].float()
    cos_row = cos_row[:, None, :].float()
    sin_row = sin_row[:, None, :].float()
    out[:, :, :half] = (x0 * cos_row - x1 * sin_row).to(out.dtype)
    out[:, :, half:rotary_dim] = (x1 * cos_row + x0 * sin_row).to(out.dtype)
    return out.reshape(num_tokens, num_heads * head_size)


def test(
    handle,
    device,
    num_tokens,
    num_q_heads,
    num_kv_heads,
    head_size,
    rotary_dim,
    sections,
    interleaved,
    position_dtype,
    dtype,
    sync=None,
):
    print(
        f"Testing MRoPE on {InfiniDeviceNames[device]} tokens={num_tokens} "
        f"q_heads={num_q_heads} kv_heads={num_kv_heads} head_size={head_size} "
        f"rotary_dim={rotary_dim} sections={sections} interleaved={interleaved} "
        f"dtype={InfiniDtypeNames[dtype]} position_dtype={InfiniDtypeNames[position_dtype]}"
    )
    q = TestTensor((num_tokens, num_q_heads * head_size), None, dtype, device)
    k = TestTensor((num_tokens, num_kv_heads * head_size), None, dtype, device)
    q_out = TestTensor(
        (num_tokens, num_q_heads * head_size), None, dtype, device, mode="zeros"
    )
    k_out = TestTensor(
        (num_tokens, num_kv_heads * head_size), None, dtype, device, mode="zeros"
    )
    max_positions = max(num_tokens + 3, 16)
    cos = TestTensor((max_positions, rotary_dim // 2), None, dtype, device)
    sin = TestTensor((max_positions, rotary_dim // 2), None, dtype, device)
    positions_torch = torch.stack(
        [
            torch.arange(num_tokens, dtype=torch.int64),
            torch.arange(num_tokens, dtype=torch.int64) + 1,
            torch.arange(num_tokens, dtype=torch.int64) + 2,
        ]
    )
    positions = TestTensor.from_torch(positions_torch, position_dtype, device)
    expected_q = torch_mrope_one(
        q.torch_tensor(),
        cos.torch_tensor(),
        sin.torch_tensor(),
        positions_torch,
        head_size,
        rotary_dim,
        sections,
        interleaved,
    )
    expected_k = torch_mrope_one(
        k.torch_tensor(),
        cos.torch_tensor(),
        sin.torch_tensor(),
        positions_torch,
        head_size,
        rotary_dim,
        sections,
        interleaved,
    )

    descriptor = infiniopOperatorDescriptor_t()
    if sync is not None:
        sync()
    check_error(
        LIBINFINIOP.infiniopCreateMRoPEDescriptor(
            handle,
            ctypes.byref(descriptor),
            q_out.descriptor,
            k_out.descriptor,
            q.descriptor,
            k.descriptor,
            cos.descriptor,
            sin.descriptor,
            positions.descriptor,
            head_size,
            rotary_dim,
            sections[0],
            sections[1],
            sections[2],
            interleaved,
        )
    )
    for tensor in [q_out, k_out, q, k, cos, sin, positions]:
        tensor.destroy_desc()
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetMRoPEWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, q.device)
    check_error(
        LIBINFINIOP.infiniopMRoPE(
            descriptor,
            workspace.data(),
            workspace_size.value,
            q_out.data(),
            k_out.data(),
            q.data(),
            k.data(),
            cos.data(),
            sin.data(),
            positions.data(),
            None,
        )
    )
    if sync is not None:
        sync()
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(q_out.actual_tensor(), expected_q, atol=atol, rtol=rtol)
        debug(k_out.actual_tensor(), expected_k, atol=atol, rtol=rtol)
    assert torch.allclose(q_out.actual_tensor(), expected_q, atol=atol, rtol=rtol)
    assert torch.allclose(k_out.actual_tensor(), expected_k, atol=atol, rtol=rtol)
    check_error(LIBINFINIOP.infiniopDestroyMRoPEDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")
