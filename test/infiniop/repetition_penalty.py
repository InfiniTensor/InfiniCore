import torch
import ctypes
from ctypes import c_uint64, c_float, POINTER, c_void_p
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
    InfiniDeviceEnum,
    infiniopOperatorDescriptor_t,
    torch_device_map,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # (num_seqs, vocab_size, penalty_value, num_tokens_to_penalize)
    (1, 512, 1.2, 10),
    (1, 4096, 1.5, 50),
    (1, 16384, 1.1, 100),
    (1, 32000, 1.3, 200),
    (2, 512, 1.2, 10),
    (4, 4096, 1.5, 50),
    (8, 16384, 1.1, 100),
    (1, 512, 0.8, 10),  # Penalty < 1.0 (reduces probability)
    (1, 4096, 2.0, 50),  # High penalty
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def apply_repetition_penalty_reference(logits, mask, repetition_penalties):
    """
    Reference implementation of repetition penalty.

    Args:
        logits: [num_seqs, vocab_size] tensor
        mask: [num_seqs, vocab_size] bool tensor
        repetition_penalties: [num_seqs] tensor of penalty values
    """
    num_seqs, vocab_size = logits.shape
    result = logits.clone()

    for seq_idx in range(num_seqs):
        penalty = repetition_penalties[seq_idx]
        for vocab_idx in range(vocab_size):
            idx = seq_idx * vocab_size + vocab_idx
            if mask[seq_idx, vocab_idx]:
                logit_val = result[seq_idx, vocab_idx].item()
                if logit_val > 0:
                    result[seq_idx, vocab_idx] = logit_val / penalty
                else:
                    result[seq_idx, vocab_idx] = logit_val * penalty

    return result


def test(
    handle,
    device,
    num_seqs,
    vocab_size,
    penalty_value,
    num_tokens_to_penalize,
    dtype=torch.float16,
    sync=None,
):
    print(
        f"Testing RepetitionPenalty on {InfiniDeviceNames[device]} with "
        f"num_seqs:{num_seqs} vocab_size:{vocab_size} penalty:{penalty_value} "
        f"num_tokens:{num_tokens_to_penalize} dtype:{InfiniDtypeNames[dtype]}"
    )

    # Create logits tensor [num_seqs, vocab_size]
    logits = TestTensor((num_seqs, vocab_size), None, dtype, device, mode="random")

    # Create mask tensor [num_seqs, vocab_size] - bool type
    # Initialize mask: set True for tokens to penalize
    # For simplicity, penalize first num_tokens_to_penalize tokens in each sequence
    mask_torch = torch.zeros((num_seqs, vocab_size), dtype=torch.bool, device=torch_device_map[device])
    for seq_idx in range(num_seqs):
        for i in range(min(num_tokens_to_penalize, vocab_size)):
            mask_torch[seq_idx, i] = True

    # Create TestTensor from the initialized mask
    mask = TestTensor.from_torch(mask_torch, InfiniDtype.BOOL, device)

    # Create repetition_penalties array [num_seqs]
    # For GPU backends, this must be a device pointer for CUDA graph compatibility
    # For CPU, host pointer is fine
    repetition_penalties = torch.full((num_seqs,), penalty_value, dtype=torch.float32)

    # Move penalties to device for GPU backends
    if device != InfiniDeviceEnum.CPU:
        repetition_penalties = repetition_penalties.to(torch_device_map[device])

    # Reference implementation
    logits_torch = logits.torch_tensor()
    expected = apply_repetition_penalty_reference(
        logits_torch, mask_torch, repetition_penalties.cpu() if device != InfiniDeviceEnum.CPU else repetition_penalties
    )

    if sync is not None:
        sync()

    # Create descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRepetitionPenaltyDescriptor(
            handle,
            ctypes.byref(descriptor),
            logits.descriptor,
            mask.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [logits, mask]:
        tensor.destroy_desc()

    # Get workspace size
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRepetitionPenaltyWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Prepare repetition_penalties array for C API
    # For GPU backends, use device pointer; for CPU, use host pointer
    if device == InfiniDeviceEnum.CPU:
        penalties_array = (c_float * num_seqs)(*repetition_penalties.cpu().numpy())
        penalties_ptr = ctypes.cast(penalties_array, POINTER(c_float))
    else:
        # For GPU, pass device pointer directly (CUDA graph compatible)
        # Cast device pointer to POINTER(c_float) for C API
        penalties_ptr = ctypes.cast(repetition_penalties.data_ptr(), POINTER(c_float))

    def lib_apply_penalty():
        check_error(
            LIBINFINIOP.infiniopApplyRepetitionPenalty(
                descriptor,
                workspace.data(),
                workspace.size(),
                logits.data(),
                mask.data(),
                penalties_ptr,
                None,
            )
        )

    lib_apply_penalty()

    if sync is not None:
        sync()

    # Compare results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(logits.actual_tensor(), expected, atol=atol, rtol=rtol)

    assert torch.allclose(
        logits.actual_tensor(), expected, atol=atol, rtol=rtol
    ), f"Repetition penalty test failed! Max diff: {torch.max(torch.abs(logits.actual_tensor() - expected)).item()}"

    # Profiling workflow
    if PROFILE:
        # Reset logits for profiling
        logits.reset()
        # fmt: off
        profile_operation(
            "PyTorch",
            lambda: apply_repetition_penalty_reference(
                logits.torch_tensor(), mask_torch, repetition_penalties
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib",
            lambda: lib_apply_penalty(),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyRepetitionPenaltyDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
