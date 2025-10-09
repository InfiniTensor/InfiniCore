from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
import ctypes
import math
import sys
import os

from regex import F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from libinfiniop import (
    open_lib,
    to_tensor,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    check_error,
    rearrange_tensor,
    create_workspace,
    get_args,
    get_test_devices,
    test_operator,
    debug,
    get_tolerance,
    profile_operation,
)

import torch


class FlashAttentionDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopFlashAttentionDescriptor_t = POINTER(FlashAttentionDescriptor)


def causal_mask(shape):
    mask = torch.tril(torch.ones(shape), diagonal=-1).flip(dims=[-2, -1])
    masked = torch.where(mask == 1, True, False)
    return masked


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
    lib,
    handle,
    torch_device,
    out_shape,
    q_shape,
    k_shape,
    v_shape,
    dtype=torch.float16,
    sync=None,
):
    print(
        f"Testing Attention on {torch_device} with out:{out_shape} q:{q_shape} k:{k_shape} v:{v_shape} dtype:{dtype}"
    )
    
    q = torch.rand(q_shape, dtype=dtype).to(torch_device) * 0.1
    k = torch.rand(k_shape, dtype=dtype).to(torch_device) * 0.1
    v = torch.rand(v_shape, dtype=dtype).to(torch_device) * 0.1
    mask = causal_mask((q_shape[-3], k_shape[-3])).to(torch_device)
    ans = attention(q, k, v, mask)
    out = torch.ones(out_shape, dtype=dtype, device=torch_device)

    out_tensor = to_tensor(out, lib)
    q_tensor = to_tensor(q, lib)
    k_tensor = to_tensor(k, lib)
    v_tensor = to_tensor(v, lib)
    # mask_ = torch.zeros((q_shape[-3], k_shape[-3]), dtype=torch.uint8).to(torch_device)
    mask_tensor = to_tensor(mask, lib)

    if sync is not None:
        sync()

    descriptor = infiniopFlashAttentionDescriptor_t()
    check_error(
        lib.infiniopCreateFlashAttentionDescriptor(
            handle,
            ctypes.byref(descriptor),
            out_tensor.descriptor,
            q_tensor.descriptor,
            k_tensor.descriptor,
            v_tensor.descriptor,
            mask_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [
        out_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        mask_tensor,
    ]:
        tensor.destroyDesc(lib)

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetFlashAttentionWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, out.device)

    def lib_attention():
        check_error(
            lib.infiniopFlashAttention(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                out_tensor.data,
                q_tensor.data,
                k_tensor.data,
                v_tensor.data,
                mask_tensor.data,
                None,
            )
        )

    lib_attention()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out, ans, atol=atol, rtol=rtol)
    assert torch.allclose(out, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: attention(q, k, v, mask), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_attention(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroyFlashAttentionDescriptor(descriptor))


if __name__ == "__main__":
    _TENSOR_DTYPES = [torch.float16]

    # Tolerance map for different data types
    _TOLERANCE_MAP = {
        torch.float16: {"atol": 1e-4, "rtol": 1e-2},
        torch.float32: {"atol": 1e-5, "rtol": 1e-3},
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
    lib = open_lib()

    lib.infiniopCreateFlashAttentionDescriptor.restype = c_int32
    lib.infiniopCreateFlashAttentionDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopFlashAttentionDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetFlashAttentionWorkspaceSize.restype = c_int32
    lib.infiniopGetFlashAttentionWorkspaceSize.argtypes = [
        infiniopFlashAttentionDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopFlashAttention.restype = c_int32
    lib.infiniopFlashAttention.argtypes = [
        infiniopFlashAttentionDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyFlashAttentionDescriptor.restype = c_int32
    lib.infiniopDestroyFlashAttentionDescriptor.argtypes = [
        infiniopFlashAttentionDescriptor_t,
    ]

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(lib, device, test, test_cases, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")