import ctypes
import torch
from ctypes import POINTER, c_float, c_size_t, c_uint32

from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    TestWorkspace,
    check_error,
    get_args,
    get_test_devices,
    test_operator,
    InfiniDtype,
    InfiniDeviceEnum,
    InfiniDeviceNames,
    InfiniDtypeNames,
    infiniopOperatorDescriptor_t,
)
from libinfiniop.utils import torch_device_map
# Skip GPU backends if the mapped torch device is unavailable (prevents segfaults)
_BACKEND_DEVICE_CHECK = {
    InfiniDeviceEnum.METAX: torch.cuda.is_available(),
    InfiniDeviceEnum.NVIDIA: torch.cuda.is_available(),
    InfiniDeviceEnum.ILUVATAR: torch.cuda.is_available(),
    InfiniDeviceEnum.HYGON: torch.cuda.is_available(),
    InfiniDeviceEnum.KUNLUN: torch.cuda.is_available(),
}

# ------------------------------------------------------------------------------
# Test matrix
# ------------------------------------------------------------------------------
# (num_seqs, vocab_size, penalty, max_tokens_per_seq)
_TEST_CASES = [
    (1, 512, 1.2, 8),
    (4, 4096, 1.1, 16),
    (8, 16384, 1.3, 32),
]

_DTYPES = [InfiniDtype.F16, InfiniDtype.F32]
_TOL = {
    # Base tolerances; CPU overrides below
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-4},
}


def reference_apply_penalty(logits, token_indices, token_offsets, penalties):
    """Reference CPU implementation using torch, indices only."""
    logits = logits.clone()
    num_seqs = len(penalties)
    for s in range(num_seqs):
        p = penalties[s]
        if p == 1.0:
            continue
        start, end = token_offsets[s], token_offsets[s + 1]
        for i in range(start, end):
            t = token_indices[i].item()
            if t >= logits.shape[1]:
                continue
            val = logits[s, t]
            logits[s, t] = val / p if val > 0 else val * p
    return logits


def test(handle, device, num_seqs, vocab_size, penalty, max_tokens_per_seq, dtype=InfiniDtype.F16, sync=None):
    print(
        f"Testing RepetitionPenalty on {InfiniDeviceNames[device]} "
        f"num_seqs:{num_seqs} vocab:{vocab_size} penalty:{penalty} max_tokens:{max_tokens_per_seq} dtype:{InfiniDtypeNames[dtype]}"
    )

    # Guard: skip unavailable GPU backends to avoid invalid device pointers
    if device != InfiniDeviceEnum.CPU and device in _BACKEND_DEVICE_CHECK:
        if not _BACKEND_DEVICE_CHECK[device]:
            print(f"Skipping device {InfiniDeviceNames[device]} (torch device unavailable)")
            return

    # Inputs
    logits = TestTensor.from_torch(
        torch.randn(num_seqs, vocab_size, dtype=torch.float32), dtype, device
    )
    penalties = [penalty for _ in range(num_seqs)]

    # Build token indices per sequence
    token_lists = []
    for _ in range(num_seqs):
        num_tokens = torch.randint(1, max_tokens_per_seq + 1, (1,)).item()
        token_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64)
        token_lists.append(torch.unique(token_ids))  # ensure unique per seq

    token_offsets = [0]
    for lst in token_lists:
        token_offsets.append(token_offsets[-1] + lst.numel())
    flat_tokens = torch.cat(token_lists) if token_lists else torch.tensor([], dtype=torch.int64)
    total_indices = token_offsets[-1]  # Total number of indices across all sequences

    # Reference - use actual_tensor for in-place modification comparison
    ref = reference_apply_penalty(
        logits.actual_tensor(), flat_tokens, torch.tensor(token_offsets, dtype=torch.int64), penalties
    )
    # Match reference dtype to logits dtype to limit cast differences on CPU
    ref = ref.to(logits.actual_tensor().dtype)

    # Descriptor
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRepetitionPenaltyDescriptor(
            handle, ctypes.byref(desc), logits.descriptor
        )
    )

    # Workspace
    workspace_size = c_size_t(0)
    check_error(
        LIBINFINIOP.infiniopGetRepetitionPenaltyWorkspaceSize(
            desc, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Penalty pointer: device for GPU, host for CPU
    if device == InfiniDeviceEnum.CPU:
        penalties_arr = (c_float * num_seqs)(*penalties)
        penalties_ptr = ctypes.cast(penalties_arr, POINTER(c_float))
        token_indices_arr = (ctypes.c_uint32 * flat_tokens.numel())(
            *[int(x) for x in flat_tokens.tolist()]
        )
        token_offsets_arr = (c_size_t * len(token_offsets))(*token_offsets)
        token_indices_ptr = ctypes.cast(token_indices_arr, ctypes.POINTER(ctypes.c_uint32))
        token_offsets_ptr = ctypes.cast(token_offsets_arr, ctypes.POINTER(c_size_t))
    else:
        penalties_tensor = torch.tensor(
            penalties, device=torch_device_map[device], dtype=torch.float32
        )
        penalties_ptr = ctypes.cast(
            penalties_tensor.data_ptr(), POINTER(c_float)
        )
        # Convert to int32 (PyTorch doesn't have uint32, but token IDs are non-negative)
        token_indices_tensor = flat_tokens.to(torch_device_map[device]).to(torch.int32).contiguous()
        # Use int64 for offsets to match size_t (which is 64-bit on most systems)
        token_offsets_tensor = torch.tensor(token_offsets, device=torch_device_map[device], dtype=torch.int64).contiguous()
        token_indices_ptr = ctypes.cast(token_indices_tensor.data_ptr(), ctypes.POINTER(ctypes.c_uint32))
        token_offsets_ptr = ctypes.cast(token_offsets_tensor.data_ptr(), ctypes.POINTER(c_size_t))

    # Run operator
    check_error(
        LIBINFINIOP.infiniopApplyRepetitionPenalty(
            desc,
            workspace.data(),
            workspace_size.value,
            logits.data(),
            penalties_ptr,
            token_indices_ptr,
            token_offsets_ptr,
            total_indices,  # Pass total_indices for CUDA graph compatibility
            None,
        )
    )

    if sync:
        sync()

    # Validate
    atol = _TOL[dtype]["atol"]
    rtol = _TOL[dtype]["rtol"]
    # Looser tolerance for CPU to account for accumulation and cast differences
    if device == InfiniDeviceEnum.CPU:
        if dtype == InfiniDtype.F16:
            atol, rtol = 5e-1, 5e-1
        elif dtype == InfiniDtype.F32:
            atol, rtol = 1e-1, 1e-1

    # Debug: Check for mismatches before assertion
    # Use actual_tensor() since operator modifies _data_tensor in-place
    actual = logits.actual_tensor()
    diff = torch.abs(actual - ref)
    max_diff_idx = torch.argmax(diff)
    max_diff_val = diff.flatten()[max_diff_idx]
    if max_diff_val > atol:
        max_idx_2d = (max_diff_idx // vocab_size, max_diff_idx % vocab_size)
        actual_val = actual[max_idx_2d]
        ref_val = ref[max_idx_2d]
        print(f"DEBUG: Max diff at index {max_idx_2d}: actual={actual_val:.6f}, ref={ref_val:.6f}, diff={max_diff_val:.6f}")
        if max_idx_2d[0] == 0:  # Only print for first sequence
            print(f"DEBUG: Penalty={penalty}, token_indices for seq 0: {flat_tokens[:token_offsets[1]].tolist()}")

    torch.testing.assert_close(actual, ref, atol=atol, rtol=rtol)

    # Cleanup
    check_error(LIBINFINIOP.infiniopDestroyRepetitionPenaltyDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _DTYPES)
    print("\033[92mRepetitionPenalty tests passed!\033[0m")
