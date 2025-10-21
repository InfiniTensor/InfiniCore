from ctypes import c_uint64
import ctypes
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

import torch




def causal_mask(shape):
    mask = torch.tril(torch.ones(shape), diagonal=-1).flip(dims=[-2, -1])
    masked = torch.where(mask == 1, True, False)
    return masked.contiguous()  # 确保返回连续张量


def attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    add_dim = False
    if(query.ndim == 3):
        query =torch.unsqueeze(query, 0)
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        add_dim = True
    B = query.size(0)
    L, S = query.size(-3), key.size(-3)
    NH, NKVH = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query.reshape(B, L, NKVH, NH//NKVH, -1).permute(0, 2, 3, 1 ,4) @ key.reshape(B, S, NKVH, 1, -1).permute(0, 2, 3, 4, 1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    attn_out = (attn_weight @ value.reshape(B, S, NKVH, 1, -1).permute(0, 2, 3, 1, 4)).permute(0, 3, 1, 2, 4).reshape(B, L, NH, -1)
    if add_dim:
        attn_out = torch.squeeze(attn_out, 0)
    return attn_out


def test(
    handle,
    device,
    out_shape,
    q_shape,
    k_shape,
    v_shape,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing FlashAttention on {InfiniDeviceNames[device]} with out:{out_shape} q:{q_shape} k:{k_shape} v:{v_shape} dtype:{InfiniDtypeNames[dtype]}"
    )
    
    # Convert InfiniDtype to torch dtype
    torch_dtype = torch.float16 if dtype == InfiniDtype.F16 else torch.float32
    
    q = TestTensor(q_shape, None, dtype, device, scale=0.1)
    k = TestTensor(k_shape, None, dtype, device, scale=0.1)
    v = TestTensor(v_shape, None, dtype, device, scale=0.1)
    mask_torch = causal_mask((q_shape[-3], k_shape[-3]))
    mask = TestTensor((q_shape[-3], k_shape[-3]), mask_torch.stride(), InfiniDtype.BOOL, device, mode="manual", set_tensor=mask_torch)
    out = TestTensor(out_shape, None, dtype, device, mode="zeros")

    # Get PyTorch tensors for reference computation
    q_torch = q.torch_tensor()
    k_torch = k.torch_tensor()
    v_torch = v.torch_tensor()
    mask_torch = mask.torch_tensor()
    ans = attention(q_torch, k_torch, v_torch, mask_torch)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFlashAttentionDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            q.descriptor,
            k.descriptor,
            v.descriptor,
            mask.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [out, q, k, v, mask]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetFlashAttentionWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_attention():
        check_error(
            LIBINFINIOP.infiniopFlashAttention(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                q.data(),
                k.data(),
                v.data(),
                mask.data(),
                None,
            )
        )

    lib_attention()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: attention(q_torch, k_torch, v_torch, mask_torch), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_attention(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyFlashAttentionDescriptor(descriptor))


if __name__ == "__main__":
    _TENSOR_DTYPES = [InfiniDtype.F16]

    # Tolerance map for different data types
    _TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-4, "rtol": 1e-2},
        InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-3},
    }

    DEBUG = False
    PROFILE = False
    NUM_PRERUN = 10
    NUM_ITERATIONS = 1000
    test_cases = [
        # basic
        # (
        #     (1, 256, 32, 64),
        #     (1, 256, 32, 64),
        #     (1, 500, 4, 64),
        #     (1, 500, 4, 64),
        # ),
        # prefill
        # (
        #     (5, 32, 64),
        #     (5, 32, 64),
        #     (5, 4, 64),
        #     (5, 4, 64),
        # ),
        (
            (15, 28, 128),
            (15, 28, 128),
            (15, 28, 128),
            (15, 28, 128),
        ),
        # decode
        # (
        #     (1, 32, 64),
        #     (1, 32, 64),
        #     (5, 4, 64),
        #     (5, 4, 64),
        # ),
    ]
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, test_cases, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")