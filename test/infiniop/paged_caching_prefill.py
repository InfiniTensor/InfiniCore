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

# ==============================================================================
# Configuration (Internal Use Only)
# ==============================================================================
_TEST_CASES = [
    # num_seqs, max_step_len, num_kv_heads, head_size, block_size, num_rounds
    (1, 16, 1, 128, 8, 5),
    (2, 64, 8, 128, 16, 2),
    (8, 128, 32, 128, 16, 3),
    (5, 512, 40, 128, 16, 3),
    (16, 64, 8, 128, 32, 1),
    (10, 256, 40, 128, 32, 3),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-8, "rtol": 1e-8},
    InfiniDtype.F16: {"atol": 1e-8, "rtol": 1e-8},
    InfiniDtype.BF16: {"atol": 1e-8, "rtol": 1e-8},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 5
NUM_ITERATIONS = 10


# ==============================================================================
# Helper Classes & Reference Implementation
# ==============================================================================
class SimpleCacheManager:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.request_to_blocks = {}
        self.request_to_len = {}

    def allocate_slots(self, request_id, num_new_tokens):
        if request_id not in self.request_to_len:
            self.request_to_len[request_id] = 0
            self.request_to_blocks[request_id] = []

        start_pos = self.request_to_len[request_id]
        new_total_len = start_pos + num_new_tokens
        needed_blocks = (new_total_len + self.block_size - 1) // self.block_size
        added_blocks = needed_blocks - len(self.request_to_blocks[request_id])

        for _ in range(added_blocks):
            self.request_to_blocks[request_id].append(self.free_blocks.pop(0))

        slots = []
        for i in range(start_pos, new_total_len):
            block_idx_in_seq = i // self.block_size
            block_offset = i % self.block_size
            physical_block_id = self.request_to_blocks[request_id][block_idx_in_seq]
            slots.append(physical_block_id * self.block_size + block_offset)

        self.request_to_len[request_id] = new_total_len
        return torch.tensor(slots, dtype=torch.int32)


def ref_paged_caching(k_new, v_new, k_pool, v_pool, slots, block_size):
    """Reference implementation for incremental caching."""
    for i in range(k_new.shape[0]):
        slot = slots[i].item()
        b_id = slot // block_size
        off = slot % block_size
        k_pool[b_id, :, off, :] = k_new[i]
        v_pool[b_id, :, off, :] = v_new[i]
    return k_pool, v_pool


# ==============================================================================
# Test Operator Implementation
# ==============================================================================
def test(
    handle,
    device,
    num_seqs,
    max_step_len,
    num_kv_heads,
    head_size,
    block_size,
    num_rounds,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing PagedCaching on {InfiniDeviceNames[device]} with "
        f"seqs:{num_seqs}, max_step_len:{max_step_len}, num_kv_heads:{num_kv_heads}, head_size:{head_size}, "
        f"block_size:{block_size}, rounds:{num_rounds}, dtype:{InfiniDtypeNames[dtype]}"
    )

    # 1. Initialize Global Cache Pool
    num_blocks = 8192
    manager = SimpleCacheManager(num_blocks, block_size)

    k_cache_pool = TestTensor(
        (num_blocks, num_kv_heads, block_size, head_size), None, dtype, device
    )
    v_cache_pool = TestTensor(
        (num_blocks, num_kv_heads, block_size, head_size), None, dtype, device
    )

    # Reference pools (CPU/Torch)
    k_pool_ref = k_cache_pool.torch_tensor().clone()
    v_pool_ref = v_cache_pool.torch_tensor().clone()

    for r in range(num_rounds):
        # Prepare incremental data for this round
        round_ntok_list = torch.randint(
            1, max_step_len + 1, (num_seqs,), dtype=torch.int32
        )
        all_slots, all_k, all_v = [], [], []

        for i in range(num_seqs):
            n_new = round_ntok_list[i].item()
            all_slots.append(manager.allocate_slots(i, n_new))
            all_k.append(torch.randn(n_new, num_kv_heads, head_size))
            all_v.append(torch.randn(n_new, num_kv_heads, head_size))

        k_in_torch = torch.cat(all_k, dim=0)
        v_in_torch = torch.cat(all_v, dim=0)
        slots_torch = torch.cat(all_slots, dim=0)

        k_in = TestTensor.from_torch(k_in_torch, dtype, device)
        v_in = TestTensor.from_torch(v_in_torch, dtype, device)
        slot_mapping = TestTensor.from_torch(slots_torch, InfiniDtype.I64, device)

        # 2. Reference Calculation
        def torch_caching():
            nonlocal k_pool_ref, v_pool_ref
            return ref_paged_caching(
                k_in.torch_tensor(),
                v_in.torch_tensor(),
                k_pool_ref,
                v_pool_ref,
                slots_torch,
                block_size,
            )

        torch_caching()

        # 3. Infiniop Operator Execution
        descriptor = infiniopOperatorDescriptor_t()
        check_error(
            LIBINFINIOP.infiniopCreatePagedCachingDescriptor(
                handle,
                ctypes.byref(descriptor),
                k_in.descriptor,
                v_in.descriptor,
                k_cache_pool.descriptor,
                v_cache_pool.descriptor,
                slot_mapping.descriptor,
            )
        )

        workspace_size = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetPagedCachingWorkspaceSize(
                descriptor, ctypes.byref(workspace_size)
            )
        )
        workspace = TestWorkspace(workspace_size.value, device)

        def lib_caching():
            check_error(
                LIBINFINIOP.infiniopPagedCaching(
                    descriptor,
                    workspace.data(),
                    workspace_size.value,
                    k_in.data(),
                    v_in.data(),
                    k_cache_pool.data(),
                    v_cache_pool.data(),
                    slot_mapping.data(),
                    None,
                )
            )

        lib_caching()
        if sync:
            sync()

        # 4. Validation
        atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

        if DEBUG:
            # Check a small slice of the updated cache
            debug(k_cache_pool.actual_tensor(), k_pool_ref, atol=atol, rtol=rtol)

        assert torch.allclose(
            k_cache_pool.actual_tensor(), k_pool_ref, atol=atol, rtol=rtol
        )
        assert torch.allclose(
            v_cache_pool.actual_tensor(), v_pool_ref, atol=atol, rtol=rtol
        )

        # 5. Profiling
        if PROFILE:
            profile_operation(
                f"Torch_R{r}",
                lambda: torch_caching(),
                device,
                NUM_PRERUN,
                NUM_ITERATIONS,
            )
            profile_operation(
                f"  Lib_R{r}", lambda: lib_caching(), device, NUM_PRERUN, NUM_ITERATIONS
            )

        check_error(LIBINFINIOP.infiniopDestroyPagedCachingDescriptor(descriptor))


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
