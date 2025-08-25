import torch
import ctypes
import gc
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

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, probs_stride, target_stride, grad_logits_stride
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
    INPLACE_PROBS = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_PROBS,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES_ = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Tolerance map for different dtypes
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

# ==============================================================================
#  Test Implementation
# ==============================================================================


def crossentropyloss_backward(grad_logits, probs, target):
    """
    PyTorch implementation of CrossEntropyLoss backward
    """
    # Calculate batch size (N) as the product of all dimensions except the last one
    batch_size = 1
    for i in range(len(probs.shape) - 1):
        batch_size *= probs.shape[i]
    
    # Compute grad_logits = (probs - target) / N
    grad_logits.copy_((probs - target) / batch_size)
    return grad_logits


def test_crossentropyloss_backward(
    handle,
    device,
    shape,
    probs_stride=None,
    target_stride=None,
    grad_logits_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float32,
    sync=None,
):
    # Convert torch dtype to InfiniDtype
    infini_dtype = InfiniDtype.F32
    if dtype == torch.float16:
        infini_dtype = InfiniDtype.F16
    elif dtype == torch.bfloat16:
        infini_dtype = InfiniDtype.BF16
    elif dtype == torch.float32:
        infini_dtype = InfiniDtype.F32
    
    # Create test tensors
    probs = TestTensor(shape, probs_stride, infini_dtype, device)
    target = TestTensor(shape, target_stride, infini_dtype, device)
    grad_logits = TestTensor(shape, grad_logits_stride, infini_dtype, device)
    
    print(
        f"Testing CrossEntropyLossBackward on {InfiniDeviceNames[device]} with shape:{shape} probs_stride:{probs_stride} target_stride:{target_stride} grad_logits_stride:{grad_logits_stride} "
        f"dtype:{InfiniDtypeNames[infini_dtype]} inplace:{inplace}"
    )

    # Initialize with random values - TestTensor already initializes with random values
    # Use softmax to generate proper probability distribution (more realistic than simple normalization)
    probs.torch_tensor().copy_(torch.softmax(probs.torch_tensor(), dim=-1))
    
    # Create proper one-hot target tensor
    # Zero out the target tensor first
    target.torch_tensor().zero_()
    # For each sample, randomly select one class to be 1 (one-hot)
    batch_shape = target.torch_tensor().shape[:-1]  # All dimensions except the last (class) dimension
    num_classes = target.torch_tensor().shape[-1]
    
    # Create random class indices for each sample in the batch
    flat_batch_size = torch.prod(torch.tensor(batch_shape)).item()
    random_indices = torch.randint(0, num_classes, (flat_batch_size,))
    
    # Set one-hot values
    target_flat = target.torch_tensor().view(flat_batch_size, num_classes)
    target_flat[torch.arange(flat_batch_size), random_indices] = 1.0

    # Set up workspace with default size 0
    workspace = TestWorkspace(0, device)

    # Create descriptor
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

    # Get workspace size
    workspace_size = ctypes.c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetCrossEntropyLossBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Ensure input data is synced to _data_tensor before library call
    probs._data_tensor.copy_(probs._torch_tensor)
    target._data_tensor.copy_(target._torch_tensor)
    
    # Define the library function
    def lib_crossentropyloss_backward():
        check_error(
            LIBINFINIOP.infiniopCrossEntropyLossBackward(
                descriptor,
                workspace.data(),
                workspace_size.value,
                grad_logits.data(),
                 probs.data(),
                 target.data(),
                None,
            )
        )

    # Run library implementation first
    lib_crossentropyloss_backward()
    
    # Sync data from device to host after library call
    if sync is not None:
        sync()
    
    # Copy the result from _data_tensor back to _torch_tensor
    grad_logits._torch_tensor.copy_(grad_logits._data_tensor)
    
    # Compute reference result using PyTorch after library call
    # Calculate batch size (N) as the product of all dimensions except the last one
    batch_size = 1
    for i in range(len(probs.torch_tensor().shape) - 1):
        batch_size *= probs.torch_tensor().shape[i]
    
    # Create a separate tensor for PyTorch reference result
    pytorch_result = (probs.torch_tensor() - target.torch_tensor()) / batch_size
    # Store the reference result in a separate variable for comparison
    reference_result = pytorch_result.clone()

    # Check results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, infini_dtype)
    if DEBUG:
        debug(grad_logits.torch_tensor(), reference_result, atol=atol, rtol=rtol)
    assert torch.allclose(grad_logits.torch_tensor(), reference_result, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: crossentropyloss_backward(grad_logits.torch_tensor(), probs.torch_tensor(), target.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_crossentropyloss_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    
    if sync is not None:
        sync()
    check_error(LIBINFINIOP.infiniopDestroyCrossEntropyLossBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Define tensor dtypes to test
    _TENSOR_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

    for device in get_test_devices(args):
        test_operator(device, test_crossentropyloss_backward, _TEST_CASES_, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")