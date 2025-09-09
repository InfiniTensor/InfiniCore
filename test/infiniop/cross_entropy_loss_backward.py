import ctypes
from ctypes import c_uint64
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
from enum import Enum, auto
import numpy as np

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# shape (N, ..., C), probs_stride, target_stride, grad_logits_stride
_TEST_CASES_ = [
    ((4, 10), None, None, None),
    ((4, 10), (10, 1), (10, 1), (10, 1)),
    ((4, 10), (0, 1), None, None),  # zero-stride broadcast on probs
    ((8, 5), None, None, None),
    ((8, 5), (10, 1), (10, 1), (10, 1)),
    ((8, 5), (5, 0), (0, 5), None),  # mixed zero-stride
    ((16, 1000), None, None, None),
    ((16, 1000), (2000, 1), (2000, 1), (2000, 1)),
    ((32, 512), None, None, None),
    ((32, 512), (1024, 1), (1024, 1), (1024, 1)),
    # Multi-dimensional cases: (N, H, W, C)
    ((2, 3, 4, 10), None, None, None),
    ((2, 3, 4, 10), (120, 40, 10, 1), (120, 40, 10, 1), (120, 40, 10, 1)),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_PROBS = auto()
    INPLACE_TARGET = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_PROBS,
    Inplace.INPLACE_TARGET,
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
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F64: {"atol": 2.25e-15, "rtol": 2.25e-15},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def cross_entropy_loss_backward_ref(
    probs: torch.Tensor, target: torch.Tensor, shape
) -> torch.Tensor:
    """Reference: grad_logits = (probs - target) / N,  N = ∏shape[:-1]."""
    shape = np.array(shape)
    batch_size = int(np.prod(shape[:-1])) if shape.size > 1 else int(shape[0])
    return (probs - target) / batch_size


def test(
    handle,
    device,
    shape,
    probs_stride=None,
    target_stride=None,
    grad_logits_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F16,
    sync=None,
):
    # Inputs
    probs = TestTensor(shape, probs_stride, dtype, device)  # 默认随机；无需 softmax
    target = TestTensor(shape, target_stride, dtype, device)  # 默认随机；无需 one-hot

    # Output / Inplace
    if inplace == Inplace.INPLACE_PROBS:
        if probs_stride != grad_logits_stride:
            return
        grad_logits = probs
    elif inplace == Inplace.INPLACE_TARGET:
        if target_stride != grad_logits_stride:
            return
        grad_logits = target
    else:
        grad_logits = TestTensor(shape, grad_logits_stride, dtype, device, mode="ones")

    if grad_logits.is_broadcast():
        return

    print(
        f"Testing CrossEntropyLossBackward on {InfiniDeviceNames[device]} with shape:{shape} "
        f"probs_stride:{probs_stride} target_stride:{target_stride} grad_logits_stride:{grad_logits_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    # Torch reference
    grad_logits._torch_tensor = cross_entropy_loss_backward_ref(
        probs.torch_tensor(), target.torch_tensor(), shape
    )

    if sync is not None:
        sync()

    # Descriptor
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

    # Workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetCrossEntropyLossBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, grad_logits.device)

    # Run lib
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

    # Check
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

    # Profile
    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: cross_entropy_loss_backward_ref(
                probs.torch_tensor(), target.torch_tensor(), shape
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib",
            lambda: lib_cross_entropy_loss_backward(),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )

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
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mCrossEntropyLoss Backward test passed!\033[0m")
