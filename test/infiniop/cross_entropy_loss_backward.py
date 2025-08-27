import ctypes
from ctypes import c_uint64
from enum import Enum, auto

import numpy as np
import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceEnum,
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

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, a_stride, b_stride, c_stride
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_A,
    Inplace.INPLACE_B,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.F64]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F64: {"atol": 2.25e-15, "rtol": 2.25e-15},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def cross_entropy_loss_backward(
    probs: torch.Tensor, target: torch.Tensor, shape
) -> torch.Tensor:
    grad_logits = probs - target
    shape = np.array(shape)
    batch_size = np.prod(shape) // shape[-1]
    grad_logits = grad_logits / batch_size
    return grad_logits


def test(
    handle,
    device,
    shape,
    probs_stride=None,
    target_stride=None,
    grad_logits_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    probs = TestTensor(shape, probs_stride, dtype, device)
    target = TestTensor(shape, target_stride, dtype, device)
    if inplace == Inplace.INPLACE_A:
        if probs_stride != grad_logits_stride:
            return
        grad_logits = probs
    elif inplace == Inplace.INPLACE_B:
        if target_stride != grad_logits_stride:
            return
        grad_logits = target
    else:
        grad_logits = TestTensor(shape, grad_logits_stride, dtype, device, mode="ones")

    if grad_logits.is_broadcast():
        return

    print(
        f"Testing CrossEntropyLossBackward on {InfiniDeviceNames[device]} with shape:{shape} probs_stride:{probs_stride} target_stride:{target_stride} grad_logits_stride:{grad_logits_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    new_grad_logits = cross_entropy_loss_backward(
        probs.torch_tensor(), target.torch_tensor(), shape
    )
    grad_logits.update_torch_tensor(new_grad_logits)

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
    for tensor in [probs, target, grad_logits]:
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
        debug(
            grad_logits.actual_tensor(),
            grad_logits.torch_tensor(),
            atol=atol,
            rtol=rtol,
        )
    assert torch.allclose(
        grad_logits.actual_tensor(), grad_logits.torch_tensor(), atol=atol, rtol=rtol
    )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: cross_entropy_loss_backward(probs.torch_tensor(), target.torch_tensor(), shape), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_cross_entropy_loss_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(
        LIBINFINIOP.infiniopDestroyCrossEntropyLossBackwardDescriptor(descriptor)
    )


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        if device == InfiniDeviceEnum.ILUVATAR:
            _TENSOR_DTYPES = [
                InfiniDtype.BF16,
                InfiniDtype.F16,
                InfiniDtype.F32,
            ]
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
