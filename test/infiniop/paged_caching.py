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
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
    TestWorkspace,
)


# ==============================================================================
#  Reference Implementation
# ==============================================================================
def ref_paged_caching(key, value, key_cache_pool, value_cache_pool, slot_mapping):
    """
    Reference implementation for paged_caching operator.

    Args:
        key (torch.Tensor): Keys, shape [ntok, nkvh, dh]
        value (torch.Tensor): Values, shape [ntok, nkvh, dh]
        key_cache_pool (torch.Tensor): K cache pool, shape [num_blocks, nkvh, block_size, dh]
        value_cache_pool (torch.Tensor): V cache pool, shape [num_blocks, nkvh, block_size, dh]
        slot_mapping (torch.Tensor): Slot mapping, shape [ntok]
    """
    ntok = key.shape[0]
    block_size = key_cache_pool.shape[2]

    # This reference implementation operates on a cloned cache to avoid modifying the original input tensor,
    # mimicking the behavior where the custom operator writes to its output tensor.
    k_cache_ref = key_cache_pool.clone()
    v_cache_ref = value_cache_pool.clone()

    for i in range(ntok):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size

        key_token = key[i]
        value_token = value[i]

        k_cache_ref[block_idx, :, block_offset, :] = key_token
        v_cache_ref[block_idx, :, block_offset, :] = value_token

    return k_cache_ref, v_cache_ref


# ==============================================================================
#  Test Configuration (Internal Use Only)
# ==============================================================================
_TEST_CASES_ = [
    # (num_seqs, max_seq_len, num_kv_heads, head_size, block_size)
    (1, 128, 8, 128, 16),
    (5, 512, 40, 128, 16),
    (16, 1024, 8, 64, 32),
    (10, 1024, 40, 64, 32),
]

# Data types for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

# Global flags for controlling test behavior
DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


def test(
    handle,
    device,
    num_seqs,  # nreq
    max_seq_len,
    num_kv_heads,  # nkvh
    head_size,  # dh
    block_size,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing PagedCaching on {InfiniDeviceNames[device]} with "
        f"num_seqs={num_seqs}, max_seq_len={max_seq_len}, num_kv_heads={num_kv_heads}, "
        f"head_size={head_size}, block_size={block_size}, dtype={InfiniDtypeNames[dtype]}"
    )

    num_blocks = 4096  # A reasonably large cache pool for testing

    # Create metadata: variable context lengths for each sequence in the batch
    context_lens_torch = torch.randint(
        1, max_seq_len + 1, (num_seqs,), dtype=torch.int32
    )
    ntok = torch.sum(context_lens_torch).item()

    # If ntok is 0 (all sequences have length 0), skip the test
    if ntok == 0:
        print("Skipping test case with ntok=0")
        return

    # Simulate the scheduler's behavior to create the slot_mapping
    slot_mapping_list = []
    current_slot = 0
    for length in context_lens_torch:
        # Find a contiguous chunk of 'length' slots
        start_slot = current_slot
        slot_mapping_list.extend(range(start_slot, start_slot + length.item()))
        current_slot += length.item()

    # Ensure we don't exceed the total number of slots in the cache
    assert (
        current_slot <= num_blocks * block_size
    ), "Not enough blocks in the cache pool for this test case"

    slot_mapping_torch = torch.tensor(slot_mapping_list, dtype=torch.int32)

    # Create input tensors based on the calculated total tokens (ntok)
    k = TestTensor((ntok, num_kv_heads, head_size), None, dtype, device)
    v = TestTensor((ntok, num_kv_heads, head_size), None, dtype, device)
    slot_mapping = TestTensor.from_torch(slot_mapping_torch, InfiniDtype.I32, device)

    # The cache pools are the "output" tensors for this operator
    k_cache_pool = TestTensor(
        (num_blocks, num_kv_heads, block_size, head_size), None, dtype, device
    )
    v_cache_pool = TestTensor(
        (num_blocks, num_kv_heads, block_size, head_size), None, dtype, device
    )

    # Run reference implementation
    k_cache_ref, v_cache_ref = ref_paged_caching(
        k.torch_tensor(),
        v.torch_tensor(),
        k_cache_pool.torch_tensor(),
        v_cache_pool.torch_tensor(),
        slot_mapping.torch_tensor(),
    )

    if sync:
        sync()

    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreatePagedCachingDescriptor(
            handle,
            ctypes.byref(descriptor),
            k.descriptor,
            v.descriptor,
            k_cache_pool.descriptor,
            v_cache_pool.descriptor,
            slot_mapping.descriptor,
        )
    )

    # Get workspace size (likely 0 for this operator, but good practice to include)
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetPagedCachingWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Invalidate descriptors to ensure kernel does not rely on them
    k.destroy_desc()
    v.destroy_desc()
    k_cache_pool.destroy_desc()
    v_cache_pool.destroy_desc()
    slot_mapping.destroy_desc()

    # Define the library call as a lambda for profiling
    def lib_paged_caching():
        check_error(
            LIBINFINIOP.infiniopPagedCaching(
                descriptor,
                workspace.data(),
                workspace_size.value,
                k.data(),
                v.data(),
                k_cache_pool.data(),
                v_cache_pool.data(),
                slot_mapping.data(),
                None,
            )
        )

    # Execute the custom operator
    lib_paged_caching()

    if sync:
        sync()

    # Verify correctness
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        print("Verifying K cache...")
        debug(k_cache_pool.actual_tensor(), k_cache_ref, atol=atol, rtol=rtol)
        print("Verifying V cache...")
        debug(v_cache_pool.actual_tensor(), v_cache_ref, atol=atol, rtol=rtol)

    assert torch.allclose(
        k_cache_pool.actual_tensor(), k_cache_ref, atol=atol, rtol=rtol
    )
    assert torch.allclose(
        v_cache_pool.actual_tensor(), v_cache_ref, atol=atol, rtol=rtol
    )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: ref_paged_caching(
            k.torch_tensor(), v.torch_tensor(), 
            k_cache_pool.torch_tensor(), v_cache_pool.torch_tensor(), 
            slot_mapping.torch_tensor()), 
            device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_paged_caching, device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    # Clean up resources
    check_error(LIBINFINIOP.infiniopDestroyPagedCachingDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options from command line arguments
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
