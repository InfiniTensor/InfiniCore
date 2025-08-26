import math
import torch
import torch.nn.functional as F
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

import warnings
warnings.filterwarnings("ignore", message=".*no current CUDA context.*")


_TEST_CASES = [
    # (
    #     (1, 2, 2, 2),                        # q/out shape
    #     (1, 2, 2, 2),                        # k/v shape
    #     infiniopAttentionMaskType.NONE,      # Mask type
    # ),
    # ((4, 2, 2), (4, 2, 2), 0),
    # ((4, 4, 4), (10, 4, 4), 0),
    # ((4, 4, 4), (4, 2, 4), 0),
    ((1, 10, 2, 4), (1, 10, 2, 4), 0),
    # ((4, 10, 8, 4), (4, 10, 2, 4), 1),
]

_TENSOR_DTYPES = [
    # InfiniDtype.F32,
    # InfiniDtype.F16,
    InfiniDtype.BF16,
]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-1, "rtol": 1e-1},
    InfiniDtype.F32: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 1e-1, "rtol": 1e-1},
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


def attention_backward(q, k, v, grad_out, attn_mask, mask_type):
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    q.grad = torch.zeros_like(q)
    k.grad = torch.zeros_like(k)
    v.grad = torch.zeros_like(v)

    if mask_type == 0:
        attn_mask = None
    elif mask_type == 2:
        attn_mask = causal_mask((q.shape[-3], k.shape[-3])).to(q.device)
        
    if q.ndim == 3:
        q_shaped = q.permute(1, 0 ,2)
        k_shaped = k.permute(1, 0 ,2)
        v_shaped = v.permute(1, 0, 2)
        grad_out_shaped = grad_out.permute(1, 0, 2)
    elif q.ndim == 4:
        q_shaped = q.permute(0, 2, 1, 3)
        k_shaped = k.permute(0, 2, 1, 3)
        v_shaped = v.permute(0, 2, 1, 3)
        grad_out_shaped = grad_out.permute(0, 2, 1, 3)
        
    out = F.scaled_dot_product_attention(
        query=q_shaped,
        key=k_shaped,
        value=v_shaped,
        attn_mask=attn_mask,
        # enable_gqa=True,
    )
    
    out.backward(grad_out_shaped)
    
    return q.grad, k.grad, v.grad


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
        f"Testing FlashAttentionBackward on {InfiniDeviceNames[device]} with qo_shape:{qo_shape} kv_shape:{kv_shape} dtype:{InfiniDtypeNames[dtype]} mask_type:{InfiniopAttentionMaskTypeNames[mask_type]}"
    )
    
    # q = torch.tensor(
    #     [[[[0.0399, 0.0517],
    #       [0.0025, 0.0940]],

    #      [[0.0946, 0.0797],
    #       [0.0415, 0.0820]]]]
    # )
    # k = torch.tensor(
    #     [[[[0.0972, 0.0791],
    #       [0.0469, 0.0330]],

    #      [[0.0334, 0.0378],
    #       [0.0764, 0.0640]]]]
    # )
    # v = torch.tensor(
    #     [[[[0.0019, 0.0773],
    #       [0.0865, 0.0810]],

    #      [[0.0667, 0.0365],
    #       [0.0364, 0.0568]]]]
    # )
    # grad_out = torch.tensor(
    #     [[[[0.0787, 0.0134],
    #       [0.0219, 0.0819]],

    #      [[0.0697, 0.0730],
    #       [0.0233, 0.0903]]]]
    # )
    # q = TestTensor.from_torch(q, dtype, device)
    # k = TestTensor.from_torch(k, dtype, device)
    # v = TestTensor.from_torch(v, dtype, device)
    # grad_out = TestTensor.from_torch(grad_out, dtype, device)
    
    grad_q = TestTensor(qo_shape, None, dtype, device, mode="zeros")
    grad_k = TestTensor(kv_shape, None, dtype, device, mode="zeros")
    grad_v = TestTensor(kv_shape, None, dtype, device, mode="zeros")
    
    q = TestTensor(qo_shape, None, dtype, device, scale=0.1)
    k = TestTensor(kv_shape, None, dtype, device, scale=0.1)
    v = TestTensor(kv_shape, None, dtype, device, scale=0.1)
    grad_out = TestTensor(qo_shape, None, dtype, device, scale=0.1)
    
    # print(f"q:\n{q.torch_tensor()}")
    # print(f"k:\n{k.torch_tensor()}")
    # print(f"v:\n{v.torch_tensor()}")
    # print(f"grad_out:\n{grad_out.torch_tensor()}")
    
    mask = causal_mask((qo_shape[-3], kv_shape[-3]))
    mask = TestTensor.from_torch(mask, InfiniDtype.F32, device)

    def torch_attention_backward():
        return attention_backward(
            q.torch_tensor(),
            k.torch_tensor(),
            v.torch_tensor(),
            grad_out.torch_tensor(),
            mask.torch_tensor(),
            mask_type
        )
        
    ans_grad_q, ans_grad_k, ans_grad_v = torch_attention_backward()
    
    if sync is not None:
        sync()
        
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFlashAttentionBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
            grad_q.descriptor,
            grad_k.descriptor,
            grad_v.descriptor,
            q.descriptor,
            k.descriptor,
            v.descriptor,
            grad_out.descriptor,
            mask.descriptor,
            mask_type,
        )
    )
    
    for tensor in [
        grad_q,
        grad_k,
        grad_v,
        q,
        k,
        v,
        grad_out,
        mask,
    ]:
        tensor.destroy_desc()
        
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetFlashAttentionBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, grad_q.device)
    
    def lib_flash_attention_backward():
        check_error(
            LIBINFINIOP.infiniopFlashAttentionBackward(
                descriptor,
                workspace.data(),
                workspace_size.value,
                grad_q.data(),
                grad_k.data(),
                grad_v.data(),
                q.data(),
                k.data(),
                v.data(),
                grad_out.data(),
                mask.data(),
                None,
            )
        )
        
    lib_flash_attention_backward()
    
    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(grad_q.actual_tensor().cpu(), ans_grad_q.cpu(), atol=atol, rtol=rtol)
        debug(grad_k.actual_tensor().cpu(), ans_grad_k.cpu(), atol=atol, rtol=rtol)
        debug(grad_v.actual_tensor().cpu(), ans_grad_v.cpu(), atol=atol, rtol=rtol)

    assert torch.allclose(grad_q.actual_tensor().cpu(), ans_grad_q.cpu(), atol=atol, rtol=rtol)
    assert torch.allclose(grad_k.actual_tensor().cpu(), ans_grad_k.cpu(), atol=atol, rtol=rtol)
    assert torch.allclose(grad_v.actual_tensor().cpu(), ans_grad_v.cpu(), atol=atol, rtol=rtol)
    
    # Profiling workflow
    if PROFILE:
        profile_operation("PyTorch", lambda: torch_attention_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_flash_attention_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
    check_error(LIBINFINIOP.infiniopDestroyFlashAttentionBackwardDescriptor(descriptor))
    
    
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
    