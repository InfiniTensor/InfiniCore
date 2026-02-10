import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TensorInitializer,
    TestCase,
    GenericTestRunner,
    is_broadcast,
)

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (shape (bs, nkvh, seq_len, hd), strides)
_TEST_CASES_DATA = [
    ((1, 1, 8, 1), None),
    ((1, 8, 32, 32), None),
    ((8, 8, 64, 32), None),
    ((1, 32, 8, 64), (32768, 1024, 64, 1)),
    ((4, 8, 32, 16), (65536, 8192, 256, 16)),
    ((8, 16, 64, 128), (8388608, 524288, 8192, 1)),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 0},
    infinicore.bfloat16: {"atol": 0, "rtol": 0},
    infinicore.float32: {"atol": 0, "rtol": 0},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []

    for data in _TEST_CASES_DATA:
        import random

        cache_shape = data[0]
        kv_shape = (
            cache_shape[0],
            cache_shape[1],
            random.randint(1, cache_shape[2]),
            cache_shape[3],
        )
        past_shape = (cache_shape[0],)

        strides = data[1]

        past_length = random.randint(0, cache_shape[2] - kv_shape[2])

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})

            cache_spec = TensorSpec.from_tensor(cache_shape, strides, dtype)
            kv_spec = TensorSpec.from_tensor(kv_shape, None, dtype)

            past_kv_lengths_spec = TensorSpec.from_tensor(
                past_shape,
                None,
                infinicore.int64,
                init_mode=TensorInitializer.RANDINT,
                low=past_length,
                high=past_length + 1,
            )

            test_cases.append(
                TestCase(
                    inputs=[
                        cache_spec,
                        cache_spec,
                        kv_spec,
                        kv_spec,
                        past_kv_lengths_spec,
                    ],
                    kwargs={},
                    output_spec=None,
                    comparison_target=[0, 1],
                    tolerance=tolerance,
                    description=f"KV Caching",
                )
            )

    return test_cases


def torch_kv_caching(k_cache, v_cache, k, v, past_kv_lengths):
    batch_size, num_kv_heads, _, head_dim = k_cache.shape
    seq_len = k.shape[2]

    for b in range(batch_size):
        past_len = past_kv_lengths[b].item()
        for h in range(num_kv_heads):
            k_cache[b, h, past_len : past_len + seq_len, :] = k[b, h, :, :]
            v_cache[b, h, past_len : past_len + seq_len, :] = v[b, h, :, :]

    return k_cache, v_cache


def infinicore_kv_caching(k_cache, v_cache, k, v, past_kv_lengths):
    infinicore.kv_caching(k_cache, v_cache, k, v, past_kv_lengths)
    return k_cache, v_cache


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("KV Caching")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_kv_caching(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore_kv_caching(*args, **kwargs)


def main():
    test_runner = GenericTestRunner(OpTest)
    test_runner.run_and_exit()


if __name__ == "__main__":
    main()
