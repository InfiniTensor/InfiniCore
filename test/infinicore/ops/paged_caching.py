import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    is_broadcast,
    TensorInitializer,
)

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (num_seqs, max_seq_len, num_kv_heads, head_size, block_size)
_TEST_CASES_DATA = [
    (1, 128, 8, 128, 16),
    (5, 512, 40, 128, 16),
    (16, 1024, 8, 64, 32),
    (10, 1024, 40, 64, 32),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-4, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


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


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects for paged_caching operation.
    Each test case contains all necessary information for execution and validation.
    """
    test_cases = []
    for num_seqs, max_seq_len, num_kv_heads, head_size, block_size in _TEST_CASES_DATA:
        num_blocks = 4096  # A reasonably large cache pool for testing

        # Create metadata: variable context lengths for each sequence in the batch
        context_lens_torch = torch.randint(
            1, max_seq_len + 1, (num_seqs,), dtype=torch.int64
        )
        ntok = torch.sum(context_lens_torch).item()

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

        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int64)

        # print("slot_mapping", slot_mapping)
        slot_mapping_shape = slot_mapping.shape

        k_shape = (ntok, num_kv_heads, head_size)
        v_shape = (ntok, num_kv_heads, head_size)
        k_cache_shape = (num_blocks, num_kv_heads, block_size, head_size)
        v_cache_shape = (num_blocks, num_kv_heads, block_size, head_size)

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            # Create typed tensor specs
            k_spec = TensorSpec.from_tensor(k_shape, None, dtype)
            v_spec = TensorSpec.from_tensor(v_shape, None, dtype)
            k_cache_spec = TensorSpec.from_tensor(k_cache_shape, None, dtype)
            v_cache_spec = TensorSpec.from_tensor(v_cache_shape, None, dtype)
            slot_mapping_spec = TensorSpec.from_tensor(
                slot_mapping_shape,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=slot_mapping,
                dtype=infinicore.int64,
            )

            # In-place operation: modifies k_cache (index 2) and v_cache (index 3)
            test_cases.append(
                TestCase(
                    inputs=[
                        k_spec,
                        v_spec,
                        k_cache_spec,
                        v_cache_spec,
                        slot_mapping_spec,
                    ],
                    kwargs=None,
                    output_spec=None,
                    comparison_target=0,  # Only compare k_cache
                    tolerance=tolerance,
                    description=f"PagedCaching",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """PagedCaching operator test with simplified implementation"""

    def __init__(self):
        super().__init__("PagedCaching")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """PyTorch paged_caching implementation"""
        return ref_paged_caching(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore paged_caching implementation"""
        return infinicore.paged_caching(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
