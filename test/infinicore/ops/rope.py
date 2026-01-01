import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import torch
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    is_broadcast,
)
from infinicore.nn.functional import RopeAlgo

import infinicore

# ==============================================================================
# Operator-specific configuration
# ==============================================================================


_TEST_CASES_DATA = [
    # bs, seq_len, num, head_dim, src strides, dst strides, Algo
    (1, 1, 1, 64, None, None, RopeAlgo.GPT_NEOX),
    (1, 5, 32, 64, None, None, RopeAlgo.GPT_NEOX),
    (1, 1, 1, 128, None, None, RopeAlgo.GPT_J),
    (1, 10, 1, 64, None, None, RopeAlgo.GPT_J),
    (2, 20, 16, 128, None, None, RopeAlgo.GPT_NEOX),
    (4, 50, 32, 256, None, None, RopeAlgo.GPT_J),
    (
        2,
        20,
        16,
        128,
        (655360, 8192, 256, 1),
        (655360, 8192, 256, 1),
        RopeAlgo.GPT_NEOX,
    ),
    (
        2,
        20,
        16,
        128,
        (655360, 8192, 256, 1),
        (655360, 8192, 256, 1),
        RopeAlgo.GPT_J,
    ),
    (
        4,
        50,
        32,
        8,
        (204800, 1024, 16, 1),
        (460800, 1536, 24, 1),
        RopeAlgo.GPT_NEOX,
    ),
    (
        4,
        50,
        32,
        8,
        (204800, 1024, 16, 1),
        (460800, 1536, 24, 1),
        RopeAlgo.GPT_J,
    ),
    (
        32,
        64,
        8,
        128,
        (1048576, 4096, 256, 1),
        (1048576, 4096, 256, 1),
        RopeAlgo.GPT_NEOX,
    ),
    (
        32,
        64,
        8,
        128,
        (1048576, 4096, 256, 1),
        (1048576, 4096, 256, 1),
        RopeAlgo.GPT_J,
    ),
    (
        64,
        17,
        32,
        64,
        (557056, 8192, 128, 1),
        (1253376, 12288, 192, 1),
        RopeAlgo.GPT_NEOX,
    ),
    (
        64,
        17,
        32,
        64,
        (557056, 8192, 128, 1),
        (1253376, 12288, 192, 1),
        RopeAlgo.GPT_J,
    ),
    (
        8,
        20,
        4,
        64,
        (1048576, 64, 262144, 1),
        (1048576, 64, 262144, 1),
        RopeAlgo.GPT_NEOX,
    ),
    (
        8,
        20,
        4,
        64,
        (1048576, 64, 262144, 1),
        (1048576, 64, 262144, 1),
        RopeAlgo.GPT_J,
    ),
    (
        8,
        20,
        32,
        64,
        (40960, 64, 1280, 1),
        (40960, 64, 1280, 1),
        RopeAlgo.GPT_NEOX,
    ),
    (
        8,
        20,
        32,
        64,
        (40960, 64, 1280, 1),
        (40960, 64, 1280, 1),
        RopeAlgo.GPT_J,
    ),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-2, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects for Rope operation.
    Each test case contains all necessary information for execution and validation.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        bs, seq_len, num, head_dim = data[0], data[1], data[2], data[3]
        src_strides, dst_strides = data[4], data[5]
        algo = data[6]

        # Determine shapes based on batch dimension
        out_shape = (bs, seq_len, num, head_dim)
        x_shape = (bs, seq_len, num, head_dim)
        sin_table_shape = (seq_len, head_dim // 2)
        cos_table_shape = (seq_len, head_dim // 2)

        # Check if tensors support in-place operations
        # x tensor supports in-place if it's not a broadcasted tensor
        x_supports_inplace = not is_broadcast(src_strides)

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            # Create typed tensor specs
            out_spec = TensorSpec.from_tensor(out_shape, dst_strides, dtype)
            x_spec = TensorSpec.from_tensor(x_shape, src_strides, dtype)
            sin_table_spec = TensorSpec.from_tensor(sin_table_shape, None, dtype)
            cos_table_spec = TensorSpec.from_tensor(cos_table_shape, None, dtype)

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[x_spec, sin_table_spec, cos_table_spec],
                    kwargs={"algo": algo},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"Rope - OUT_OF_PLACE",
                )
            )

            # Test Case 2: In-place with explicit output tensor
            if dst_strides is None or not is_broadcast(dst_strides):
                test_cases.append(
                    TestCase(
                        inputs=[x_spec, sin_table_spec, cos_table_spec],
                        kwargs={"algo": algo},
                        output_spec=out_spec,  # Specify the output tensor spec
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"Rope - INPLACE(out)",
                    )
                )

            # Test Case 3: In-place on input tensor (x)
            if x_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[x_spec, sin_table_spec, cos_table_spec],
                        kwargs={"algo": algo, "out": 0},  # Use index 0 for first input
                        output_spec=None,
                        comparison_target=0,  # Compare first input (x tensor)
                        tolerance=tolerance,
                        description=f"Rope - INPLACE(x)",
                    )
                )

    return test_cases


def rotary_embedding(t, sin, cos, algo, *, out=None):
    def _torch_rope(sin, cos, t1, t2):
        cos = cos.unsqueeze(1)  # [seq_len, 1, dh // 2]
        sin = sin.unsqueeze(1)  # [seq_len, 1, dh // 2]
        t_out_1 = t1 * cos - t2 * sin
        t_out_2 = t1 * sin + t2 * cos

        return t_out_1, t_out_2

    # If out parameter is provided and it's the same as input t, operate in-place
    if out is not None:
        if out.data_ptr() == t.data_ptr():
            ans = t  # Use the same tensor for in-place operation
        else:
            ans = out  # Use provided output tensor
    else:
        ans = t.clone()

    dh = t.shape[-1]
    dt = t.dtype
    assert dh % 2 == 0, "Embedding dimension must be even."

    if RopeAlgo.GPT_J == algo:
        t_even = t[..., 0::2]  # [bs, seq_len, n_head, dh // 2]
        t_odd = t[..., 1::2]  # [bs, seq_len, n_head, dh // 2]

        t_out_even, t_out_odd = _torch_rope(sin, cos, t_even, t_odd)

        ans[..., 0::2] = t_out_even.to(dt)
        ans[..., 1::2] = t_out_odd.to(dt)
    elif RopeAlgo.GPT_NEOX == algo:
        half_dim = dh // 2
        t_first = t[..., :half_dim]
        t_second = t[..., half_dim:]

        t_out_first, t_out_second = _torch_rope(sin, cos, t_first, t_second)

        ans[..., :half_dim] = t_out_first.to(dt)
        ans[..., half_dim:] = t_out_second.to(dt)
    else:
        raise KeyError("Unsupported RoPE algorithm")

    # If operating in-place on t, we don't need to copy back
    if out is not None and out.data_ptr() != t.data_ptr():
        out.copy_(ans)
        return out
    return ans


class OpTest(BaseOperatorTest):
    """Rope operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Rope")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """PyTorch Rope implementation"""

        return rotary_embedding(*args, **kwargs)

    def infinicore_operator(self, x, sin_table, cos_table, algo, out=None, **kwargs):
        """InfiniCore Rope implementation"""

        bs, seq_len, num, head_dim = x.shape

        infini_device = x.device
        torch_device = torch.device(type=infini_device.type, index=infini_device.index)

        ## 创建 pos_ids的变量
        cache_position_list = [list(range(0, seq_len)) for i in range(bs)]
        pos_ids_torch = torch.tensor(cache_position_list, device=torch_device)
        pos_ids_infini = infinicore.from_torch(pos_ids_torch)

        # 计算
        pos_ids = pos_ids_infini
        return infinicore.nn.functional.rope(
            x, pos_ids, sin_table, cos_table, algo=algo, out=out
        )


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
