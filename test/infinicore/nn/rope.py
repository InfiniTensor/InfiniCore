import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from infinicore.nn.functional import RopeAlgo

import infinicore

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (x_shape)
# bs, seq_len, num_heads, head_dim
_TEST_CASES_DATA = [
    # Basic cases
    (1, 10, 32, 64),
    (2, 2, 32, 64),
    (5, 10, 32, 64),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 5e-2, "rtol": 5e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects for all operation types.
    Each test case contains all necessary information for execution and validation.
    """
    test_cases = []

    for bs, seq_len, num_heads, head_dim in _TEST_CASES_DATA:
        strides = None

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            x_shape = [bs, seq_len, num_heads, head_dim]

            # Create typed tensor specs
            x_spec = TensorSpec.from_tensor(x_shape, strides, dtype, name="x")

            max_position_embeddings = 1024
            rope_theta = 10000.0

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[x_spec],
                    kwargs={
                        "max_position_embeddings": max_position_embeddings,
                        "rope_theta": rope_theta,
                    },
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"nn.RoPE - OUT_OF_PLACE",
                )
            )

    return test_cases


def rotary_embedding(
    t,
    max_position_embeddings,
    rope_theta,
    head_dim,
    algo=RopeAlgo.GPT_NEOX,
):
    def create_sin_cos_table(
        max_position,
        head_dim,
        theta=10000.0,
        torch_dtype=torch.float32,
        torch_device="cpu",
    ):
        assert head_dim % 2 == 0, "Embedding dimension must be even."
        pos = torch.arange(0, max_position)
        freqs = 1.0 / (
            theta
            ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
        )
        angles = torch.outer(pos, freqs)
        return torch.sin(angles).to(dtype=torch_dtype, device=torch_device), torch.cos(
            angles
        ).to(dtype=torch_dtype, device=torch_device)

    def _torch_rope(sin, cos, t1, t2):
        cos = cos.unsqueeze(1)  # [seq_len, 1, dh // 2]
        sin = sin.unsqueeze(1)  # [seq_len, 1, dh // 2]
        t_out_1 = t1 * cos - t2 * sin
        t_out_2 = t1 * sin + t2 * cos

        return t_out_1, t_out_2

    sin, cos = create_sin_cos_table(
        max_position_embeddings, head_dim, rope_theta, torch_device=t.device
    )

    ans = t.clone()
    dh = t.shape[-1]
    dt = t.dtype
    assert dh % 2 == 0, "Embedding dimension must be even."

    if RopeAlgo.GPT_J == algo:
        t_even = t[..., 0::2]  # [seq_len, n_head, dh // 2]
        t_odd = t[..., 1::2]  # [seq_len, n_head, dh // 2]

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
        raise KeyError("error Algo ")

    return ans


class OpTest(BaseOperatorTest):
    """nn.RoPE test with simplified implementation"""

    def __init__(self):
        super().__init__("nn.RoPE")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, max_position_embeddings, rope_theta):
        """PyTorch nn.RoPE implementation"""

        bs, seq_len, num_heads, head_dim = x.shape

        return rotary_embedding(x, seq_len, rope_theta, head_dim)

    def infinicore_operator(self, x, max_position_embeddings, rope_theta):
        """InfiniCore nn.RoPE implementation"""

        bs, seq_len, num_heads, head_dim = x.shape
        torch_device = "cpu"
        if x.device.type != "cpu":
            torch_device = "cuda"

        # 创建 pos_ids的变量
        pos_ids_torch = torch.arange(0, seq_len, dtype=torch.int32, device=torch_device)
        pos_ids_torch = pos_ids_torch.unsqueeze(0)
        pos_ids_torch = pos_ids_torch.expand(bs, seq_len).contiguous()

        pos_ids_infini = infinicore.from_torch(pos_ids_torch)

        # 创建类
        rope_instance = infinicore.nn.RoPE(
            max_position_embeddings,
            rope_theta,
            head_dim,
            device=x.device,
            dtype=x.dtype,
        )

        # 计算
        y = rope_instance(x, pos_ids_infini)
        return y


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
