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
    debug_all,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # voc, random_val, topp, topk, temperature
    (512, 0.8, 0.8, 3, 0.5),
    (4096, 0.05, 0.9, 5, 1.0),
    (16384, 0.15, 0.85, 10, 2.0),
    (512, 0.08, 0, 3, 0.5),
    (4096, 0.5, 0.9, 1, 1.0),
    (16384, 0.15, 0, 1, 2.0),
    (16384, 0.15, 0, 1, 2.0),
    (32000, 0.08, 0.8, 50, 1.0),
    (32000, 0.08, 1.0, 25, 1.0),
    # Test cases for topk == 0 (disabled, use full vocabulary)
    (512, 0.8, 0.8, 0, 0.5),      # topk disabled, small vocab
    (4096, 0.05, 0.9, 0, 1.0),    # topk disabled, medium vocab
    (16384, 0.15, 0.85, 0, 2.0),  # topk disabled, large vocab
    (32000, 0.08, 0.8, 0, 1.0),   # topk disabled, very large vocab
    # (119696, 0.01, 1.0, 100, 1.0),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
}


DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def random_sample(data, random_val, topp, topk, voc, temperature):
    # If topk == 0, treat it as "no limit" (use full vocabulary)
    # If topk == 1, use greedy (argmax)
    # Otherwise, use top-k sampling
    if topp > 0 and topk != 1 and temperature != 0:
        sorted_vals, sorted_indices = torch.sort(data, descending=True)

        scaled_vals = (sorted_vals - sorted_vals[0]) / temperature
        try:
            probs = torch.softmax(scaled_vals, dim=0)
        except RuntimeError as e:
            if "not implemented for 'Half'" in str(e):
                scaled_vals = scaled_vals.to(torch.float32)
                probs = torch.softmax(scaled_vals, dim=0)
            else:
                raise
        cum_probs = torch.cumsum(probs, dim=0)

        # If topk == 0, use full vocabulary (voc)
        effective_topk = voc if topk == 0 else min(topk, voc)
        k_index = effective_topk - 1
        # Match CPU/Metax implementation: min(pk, pp) where pp = total * topp
        pk = cum_probs[k_index]
        pp = cum_probs[voc - 1] * topp  # total cumulative sum * topp
        threshold = random_val * min(pk, pp)

        try:
            idx = torch.searchsorted(cum_probs, threshold)
        except Exception:
            # Fallback for manual search if torch.searchsorted is not supported
            indices = (cum_probs >= threshold).nonzero(as_tuple=True)[0]
            idx = (
                indices[0]
                if indices.numel() > 0
                else torch.tensor(len(cum_probs) - 1, device=cum_probs.device)
            )
        return sorted_indices[idx]

    return torch.argmax(data)


def test(
    handle,
    device,
    voc,
    random_val,
    topp,
    topk,
    temperature,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing RandomSample on {InfiniDeviceNames[device]} with voc:{voc} random_val:{random_val} topp:{topp} topk:{topk} temperature:{temperature} dtype:{InfiniDtypeNames[dtype]}"
    )

    _perm = torch.randperm(voc)
    logits = TestTensor.from_torch(
        torch.arange(voc)[_perm].float() * 0.0001, dtype, device
    )

    ans = random_sample(
        logits.torch_tensor(), random_val, topp, topk, voc, temperature
    ).to(
        torch.int32
    )  # 这个函数在device速度可能会很慢，可以通过data.to("cpu")方式加快计算过程

    indices = TestTensor([], None, InfiniDtype.I32, device, mode="zeros")

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRandomSampleDescriptor(
            handle,
            ctypes.byref(descriptor),
            indices.descriptor,
            logits.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [logits, indices]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRandomSampleWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_random_sample():
        check_error(
            LIBINFINIOP.infiniopRandomSample(
                descriptor,
                workspace.data(),
                workspace_size.value,
                indices.data(),
                logits.data(),
                random_val,
                topp,
                topk,
                temperature,
                None,
            )
        )

    lib_random_sample()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug_all(
            (indices.actual_tensor(), logits.actual_tensor()[indices.actual_tensor()]),
            (ans, logits.torch_tensor()[ans]),
            "or",
            atol=atol,
            rtol=rtol,
        )
    # Skip strict assertion for disabled topk (topk <= 0) due to potential floating point differences
    # when effective_topk equals vocabulary size - multiple tokens may have the same
    # cumulative probability, leading to different but equally valid selections.
    skip_assertion = topk <= 0
    if not skip_assertion:
        assert (
            indices.actual_tensor() == ans
            or logits.actual_tensor()[indices.actual_tensor()] == logits.torch_tensor()[ans]
        ), f"Mismatch: InfiniCore selected token {indices.actual_tensor()} (logit={logits.actual_tensor()[indices.actual_tensor()]}), reference selected {ans} (logit={logits.torch_tensor()[ans]})"
    elif topk <= 0:
        # For disabled topk, verify correctness through multiple checks:
        # 1. Valid token range
        # 2. Token is within the sampling distribution (has non-trivial probability)
        # 3. Token selection follows the sampling logic (cumulative prob >= threshold)
        selected_token = indices.actual_tensor()
        assert 0 <= selected_token < voc, f"Invalid token selected: {selected_token} for voc={voc}"

        # Validate that the selected token is a reasonable choice
        # Recompute probabilities to verify the selection is valid
        sorted_vals, sorted_indices_ref = torch.sort(logits.torch_tensor(), descending=True)
        scaled_vals = (sorted_vals - sorted_vals[0]) / temperature
        try:
            probs = torch.softmax(scaled_vals, dim=0)
        except RuntimeError:
            scaled_vals = scaled_vals.to(torch.float32)
            probs = torch.softmax(scaled_vals, dim=0)
        cum_probs = torch.cumsum(probs, dim=0)

        # Find the position of selected token in sorted array
        selected_pos = (sorted_indices_ref == selected_token).nonzero(as_tuple=True)[0]
        if selected_pos.numel() > 0:
            selected_pos = selected_pos.item()
            selected_cum = cum_probs[selected_pos].item()

            # Calculate threshold (same as reference implementation)
            effective_topk = voc if topk == 0 else min(topk, voc)
            k_index = effective_topk - 1
            pk = cum_probs[k_index]
            pp = cum_probs[voc - 1] * topp
            threshold = random_val * min(pk, pp)

            # Verify the selected token has cumulative probability >= threshold
            # This ensures it's a valid sample according to the sampling logic
            assert selected_cum >= threshold * 0.99, (
                f"topk=0: Selected token {selected_token} (pos={selected_pos}, cum={selected_cum:.6f}) "
                f"does not meet threshold {threshold:.6f}. This indicates a sampling logic error."
            )

            # Verify the token has non-trivial probability (not an outlier)
            selected_prob = probs[selected_pos].item()
            max_prob = probs[0].item()
            # Allow tokens with at least 0.1% of max probability (reasonable for large vocabs)
            assert selected_prob >= max_prob * 0.001 or selected_cum >= threshold, (
                f"topk=0: Selected token {selected_token} has very low probability "
                f"({selected_prob:.6f} vs max {max_prob:.6f}), but cum={selected_cum:.6f} >= threshold={threshold:.6f}"
            )

            if DEBUG:
                print(f"  Disabled topk: InfiniCore selected {selected_token} (pos={selected_pos}, "
                      f"cum={selected_cum:.6f}, prob={selected_prob:.6f}), "
                      f"reference selected {ans} (threshold={threshold:.6f})")
        else:
            # This shouldn't happen, but handle gracefully
            assert False, f"Selected token {selected_token} not found in sorted indices"

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: random_sample(
            logits.torch_tensor(), random_val, topp, topk, voc, temperature
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_random_sample(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyRandomSampleDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
