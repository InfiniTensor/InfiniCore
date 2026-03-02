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
def torch_kv_caching(k_cache, v_cache, k, v, past_kv_lengths):
    #k_cache.shape=[batch_size, num_kv_heads, max_seq_len, hidden_dim]
    #v_cache.shape=[batch_size, num_kv_heads, max_seq_len, hidden_dim]
    #k.shape=[batch_size, num_kv_heads, seq_len, hidden_dim]
    #v.shape=[batch_size, num_kv_heads, seq_len, hidden_dim]
    #past_kv_lengths.shape = [batch_size]
    batch_size, num_kv_heads, _, head_dim = k_cache.shape
    seq_len = k.shape[2]

    for b in range(batch_size):
        past_len = past_kv_lengths[b].item()
        for h in range(num_kv_heads):
            k_cache[b, h, past_len : past_len + seq_len, :] = k[b, h, :, :]
            v_cache[b, h, past_len : past_len + seq_len, :] = v[b, h, :, :]

    return k_cache, v_cache


# ==============================================================================
#  Test Configuration (Internal Use Only)
# ==============================================================================
_TEST_CASES_ = [
    # (num_seqs, num_kv_heads, max_seq_len, hidden_dim), strides
    ((1, 1, 8, 1), None),
    ((1, 8, 32, 32), None),
    ((8, 8, 64, 32), None),
    ((1, 32, 8, 64), (32768, 1024, 64, 1)),
    ((4, 8, 32, 16), (65536, 8192, 256, 16)),
    ((8, 16, 64, 128), (8388608, 524288, 8192, 1)),
]

# Data types for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
    InfiniDtype.F32: {"atol": 0, "rtol": 0},
}

# Global flags for controlling test behavior
DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


def test(
    handle,
    device,
    cache_shape,
    strides,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing KVCaching on {InfiniDeviceNames[device]} with cache_shape:{cache_shape}, strides:{strides}, dtype={InfiniDtypeNames[dtype]}"
    )

    import random

    kv_shape = (
            cache_shape[0],
            cache_shape[1],
            random.randrange(1, cache_shape[2]),
            cache_shape[3],
        )
    past_shape = (cache_shape[0],)
    
    k_cache = TestTensor(cache_shape, strides, dtype, device)
    v_cache = TestTensor(cache_shape, strides, dtype, device)

    k = TestTensor(kv_shape, None, dtype, device)
    v = TestTensor(kv_shape, None, dtype, device)

    past_kv_lengths = TestTensor(past_shape, None, InfiniDtype.I64, device, randint_low=0, randint_high=cache_shape[2] - kv_shape[2])
   
    # Run reference implementation
    k_cache_ref, v_cache_ref = torch_kv_caching(
        k_cache.torch_tensor(), 
        v_cache.torch_tensor(), 
        k.torch_tensor(), 
        v.torch_tensor(), 
        past_kv_lengths.torch_tensor())

    if sync:
        sync()

    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateKVCachingDescriptor(
            handle,
            ctypes.byref(descriptor),
            k_cache.descriptor,
            v_cache.descriptor,
            k.descriptor,
            v.descriptor,
            past_kv_lengths.descriptor,
        )
    )

    # Get workspace size (likely 0 for this operator, but good practice to include)
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetKVCachingWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Invalidate descriptors to ensure kernel does not rely on them
    k.destroy_desc()
    v.destroy_desc()
    k_cache.destroy_desc()
    v_cache.destroy_desc()
    past_kv_lengths.destroy_desc()

    # Define the library call as a lambda for profiling
    def lib_kv_caching():
        check_error(
            LIBINFINIOP.infiniopKVCaching(
                descriptor,
                workspace.data(),
                workspace_size.value,
                k_cache.data(),
                v_cache.data(),
                k.data(),
                v.data(),
                past_kv_lengths.data(),
                None,
            )
        )

    # Execute the custom operator
    lib_kv_caching()

    if sync:
        sync()

    # Verify correctness
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        print("Verifying K cache...")
        debug(k_cache.actual_tensor(), k_cache_ref, atol=atol, rtol=rtol)
        print("Verifying V cache...")
        debug(v_cache.actual_tensor(), v_cache_ref, atol=atol, rtol=rtol)

    assert torch.allclose(
        k_cache.actual_tensor(), k_cache_ref, atol=atol, rtol=rtol
    )
    assert torch.allclose(
        v_cache.actual_tensor(), v_cache_ref, atol=atol, rtol=rtol
    )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_kv_caching(k_cache.torch_tensor(), v_cache.torch_tensor(), k.torch_tensor(), v.torch_tensor(), past_kv_lengths.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_kv_caching, device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    # Clean up resources
    check_error(LIBINFINIOP.infiniopDestroyKVCachingDescriptor(descriptor))


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
