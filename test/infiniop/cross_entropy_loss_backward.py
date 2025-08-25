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
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto

_TEST_CASES_ = [
    # shape, grad_logits_stride, probs_stride, target_stride
    ((13, 4), None, None, None),
    ((13, 4), (1, 13), (1, 13), (1, 13)),
    ((16, 5632), None, None, None),
    ((16, 5632), (1, 16), (1, 16), (1, 16)),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]


# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types


_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_cross_entropy_loss_backward(grad_logits, logits:torch.Tensor, probs, target):
    logits.requires_grad_(True)
    loss = torch.nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    grad_logits.copy_(logits.grad)
    logits.requires_grad_(False)
    
    

def test(
    handle,
    device,
    input_shape,
    grad_logits_stride,
    probs_stride,
    target_stride,
    inplace,
    dtype,
    sync=None,
):
    print(
        f"Testing cross_entropy_loss_backward on {InfiniDeviceNames[device]} with shape:{input_shape},"
        f"grad_logits_stride:{grad_logits_stride} grad_logits_stride:{grad_logits_stride}, target_stride:{target_stride},"
        f"inplace:{inplace},"  
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    logits = TestTensor(
        input_shape,
        None,
        dtype,
        device,
    )
    torch_logits = logits.torch_tensor()
    torch_probs = torch.softmax(torch_logits, dim=-1)
    # if probs_stride is not None:
    #     torch_probs.as_strided_(input_shape, probs_stride)
    probs = TestTensor(
        input_shape,
        torch_probs.stride(),
        dtype,
        device,
        "manual",
        set_tensor=torch_probs
    )

    if inplace == Inplace.INPLACE:
        if grad_logits_stride != probs_stride:
            return
        grad_logits = probs
    else:
        grad_logits = TestTensor(
            input_shape,
            grad_logits_stride,
            dtype,
            device,
        )

    def get_batch_size(shape):
        batch_size = 1
        for d in range(len(shape) - 1):
            batch_size *= shape[d]  
        return batch_size      
    batch_size = get_batch_size(input_shape)

    torch_target = torch.where(
        torch.arange(input_shape[-1]).unsqueeze(0).repeat(batch_size, 1) == \
        torch.randint(low=0, high=input_shape[-1], size=(batch_size,)).unsqueeze(1).repeat(1, input_shape[-1]),
        1, 0
    ).reshape(input_shape).type(torch_logits.dtype)
    # if target_stride is not None:
    #     torch_target.as_strided_(input_shape, target_stride)

    target = TestTensor(
        input_shape,
        torch_target.stride(),
        dtype,
        device,
        "manual",
        set_tensor=torch_target
    )

    torch_cross_entropy_loss_backward(
        grad_logits.torch_tensor(), torch_logits, probs.torch_tensor(), target.torch_tensor()
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateCrossEntropyLossBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
            grad_logits.descriptor,
            probs.descriptor,
            target.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [grad_logits, probs, target]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetCrossEntropyLossBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, grad_logits.device)

    def lib_cross_entropy_loss_backward():
        check_error(
            LIBINFINIOP.infiniopCrossEntropyLossBackward(
                descriptor,
                workspace.data(),
                workspace.size(),
                grad_logits.data(),
                probs.data(),
                target.data(),                
                None,
            )
        )

    lib_cross_entropy_loss_backward()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(grad_logits.actual_tensor(), grad_logits.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(grad_logits.actual_tensor(), grad_logits.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_cross_entropy_loss_backward(
            grad_logits.torch_tensor(), torch_logits, probs.torch_tensor(), target.torch_tensor()
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_cross_entropy_loss_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyCrossEntropyLossBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my cross_entropy_loss_backward passed!\033[0m")
