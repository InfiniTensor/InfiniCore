import os
import sys
from ctypes import byref, c_char, c_float, c_uint64

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
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
    profile_operation,
    test_operator,
)


def torch_flash_attention(q, k, v, total_kv_len, scale, is_causal):
    batch, num_q_heads, query_len, _ = q.shape
    num_kv_heads = k.shape[1]
    assert num_q_heads % num_kv_heads == 0

    outs = []
    lengths = total_kv_len.detach().cpu().tolist()
    for b in range(batch):
        kv_len = min(int(lengths[b]), k.shape[2])
        q_b = q[b : b + 1].float()
        k_b = k[b : b + 1, :, :kv_len, :].float()
        v_b = v[b : b + 1, :, :kv_len, :].float()
        if num_q_heads != num_kv_heads:
            group_size = num_q_heads // num_kv_heads
            k_b = k_b.repeat_interleave(group_size, dim=1)
            v_b = v_b.repeat_interleave(group_size, dim=1)

        logits = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale
        if is_causal:
            first_query_key = max(kv_len - query_len, 0)
            q_pos = torch.arange(query_len, device=q.device)[:, None]
            k_pos = torch.arange(kv_len, device=q.device)[None, :]
            mask = k_pos <= (first_query_key + q_pos)
            logits = logits.masked_fill(
                ~mask.view(1, 1, query_len, kv_len), float("-inf")
            )

        outs.append(torch.matmul(torch.softmax(logits, dim=-1), v_b).to(q.dtype))

    return torch.cat(outs, dim=0)


def test(
    handle,
    device,
    q_shape,
    k_shape,
    v_shape,
    total_kv_len_values,
    is_causal,
    q_stride=None,
    k_stride=None,
    v_stride=None,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing FlashAttention on {InfiniDeviceNames[device]} "
        f"q:{q_shape} k:{k_shape} v:{v_shape} dtype:{InfiniDtypeNames[dtype]} "
        f"is_causal:{is_causal} q_stride:{q_stride} k_stride:{k_stride} v_stride:{v_stride}"
    )

    scale = q_shape[-1] ** -0.5
    out = TestTensor(q_shape, None, dtype, device, mode="zeros")
    q = TestTensor(q_shape, q_stride, dtype, device, scale=0.1)
    k = TestTensor(k_shape, k_stride, dtype, device, scale=0.1)
    v = TestTensor(v_shape, v_stride, dtype, device, scale=0.1)
    total_kv_len = TestTensor.from_torch(
        torch.tensor(total_kv_len_values, dtype=torch.int64),
        InfiniDtype.I64,
        device,
    )

    def torch_op():
        return torch_flash_attention(
            q.torch_tensor(),
            k.torch_tensor(),
            v.torch_tensor(),
            total_kv_len.torch_tensor(),
            scale,
            is_causal,
        )

    ans = torch_op()

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFlashAttentionDescriptor(
            handle,
            byref(descriptor),
            out.descriptor,
            q.descriptor,
            k.descriptor,
            v.descriptor,
            total_kv_len.descriptor,
            c_float(scale),
            c_char(b"\x01" if is_causal else b"\x00"),
        )
    )

    for tensor in [out, q, k, v, total_kv_len]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetFlashAttentionWorkspaceSize(
            descriptor, byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, out.device)

    def lib_op():
        check_error(
            LIBINFINIOP.infiniopFlashAttention(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                q.data(),
                k.data(),
                v.data(),
                total_kv_len.data(),
                None,
            )
        )

    lib_op()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation("PyTorch", torch_op, device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_op, device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyFlashAttentionDescriptor(descriptor))


if __name__ == "__main__":
    _TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]
    _TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},
        InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
        InfiniDtype.F32: {"atol": 1e-3, "rtol": 1e-3},
    }

    DEBUG = False
    PROFILE = False
    NUM_PRERUN = 10
    NUM_ITERATIONS = 1000

    test_cases = [
        (
            (1, 16, 1, 128),
            (1, 8, 256, 128),
            (1, 8, 256, 128),
            [256],
            True,
            None,
            None,
            None,
        ),
        (
            (1, 16, 16, 128),
            (1, 8, 256, 128),
            (1, 8, 256, 128),
            [256],
            True,
            None,
            None,
            None,
        ),
        ((2, 4, 3, 16), (2, 2, 8, 16), (2, 2, 8, 16), [8, 6], True, None, None, None),
        (
            (1, 28, 64, 128),
            (1, 4, 512, 128),
            (1, 4, 512, 128),
            [512],
            True,
            None,
            None,
            None,
        ),
        ((1, 8, 4, 16), (1, 8, 64, 16), (1, 8, 64, 16), [52], False, None, None, None),
    ]

    args = get_args()
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, test_cases, _TENSOR_DTYPES)
