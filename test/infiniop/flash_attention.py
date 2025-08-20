import math
import torch
import ctypes
from ctypes import c_uint64
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
    InfiniDeviceEnum,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
    infiniopAttentionMaskType,
    InfiniopAttentionMaskTypeNames,
)


_TEST_CASES = [
    # (
    #     (1, 2, 2, 2),                        # q/out shape
    #     (1, 2, 2, 2),                        # k/v shape
    #     infiniopAttentionMaskType.NONE,      # Mask type
    # ),
    ((4, 2, 2), (4, 2, 2), 0),
    ((4, 4, 4), (10, 4, 4), 0),
    ((10, 4, 4), (4, 4, 4), 2),
    ((1, 10, 2, 4), (1, 10, 2, 4), 0),
    ((4, 10, 8, 4), (4, 10, 2, 4), 1),
]

_TENSOR_DTYPES = [
    InfiniDtype.F32,
    InfiniDtype.F16,
    InfiniDtype.BF16,
]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def causal_mask(shape):
    mask = torch.tril(torch.ones(shape, dtype=torch.float32))
    mask = torch.where(mask == 1, 
                      torch.tensor(0.0, dtype=torch.float32), 
                      torch.tensor(float('-inf'), dtype=torch.float32))
    return mask


def attention(query, key, value, attn_mask=None, mask_type=None, dropout_p=0.0, scale=None) -> torch.Tensor:
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
    
    if mask_type == 0:
        attn_mask = None
    elif mask_type == 2:
        attn_mask = causal_mask((L, S)).to(query.device)

    if attn_mask is not None:
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
    qo_shape,
    kv_shape,
    mask_type,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing FlashAttention on {InfiniDeviceNames[device]} with qo_shape:{qo_shape} kv_shape:{kv_shape}  dtype:{InfiniDtypeNames[dtype]} mask_type:{InfiniopAttentionMaskTypeNames[mask_type]}"
    )
    
    out = TestTensor(qo_shape, None, dtype, device, mode="zeros")
    l = TestTensor(qo_shape[:-1], None, dtype, device, mode="zeros")
    
    q = TestTensor(qo_shape, None, dtype, device, scale=0.1)
    k = TestTensor(kv_shape, None, dtype, device, scale=0.1)
    v = TestTensor(kv_shape, None, dtype, device, scale=0.1)
        
    mask = causal_mask((qo_shape[-3], kv_shape[-3]))
    mask = TestTensor.from_torch(mask, InfiniDtype.F32, device)
    
    def torch_attention():
        return attention(
            q.torch_tensor(),
            k.torch_tensor(),
            v.torch_tensor(),
            mask.torch_tensor(),
            mask_type,
        )

    ans = torch_attention()
    
    if sync is not None:
        sync()
        
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFlashAttentionDescriptor(
            handle, 
            ctypes.byref(descriptor), 
            out.descriptor, 
            l.descriptor,
            q.descriptor, 
            k.descriptor, 
            v.descriptor,
            mask.descriptor,
            mask_type,
        )
    )
    
    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [
        out,
        l,
        q,
        k,
        v,
        mask,
    ]:
        tensor.destroy_desc()
    
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetFlashAttentionWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, out.device)
    
    def lib_flash_attention():
        check_error(
            LIBINFINIOP.infiniopFlashAttention(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                l.data(),
                q.data(),
                k.data(),
                v.data(),
                mask.data(),
                None,
            )
        )
    
    lib_flash_attention()
    
    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        profile_operation("PyTorch", lambda: torch_attention(), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_flash_attention(), device, NUM_PRERUN, NUM_ITERATIONS)
    check_error(LIBINFINIOP.infiniopDestroyFlashAttentionDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    
    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
        
    print("\033[92mTest passed!\033[0m")
    