import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)

import infinicore

_TEST_CASES_DATA = [
    # num_tokens, num_q_heads, num_kv_heads, head_size, rotary_dim, sections, interleaved
    (5, 2, 1, 32, 32, (4, 6, 6), False),
    (7, 3, 1, 32, 24, (2, 4, 6), False),
    (6, 2, 2, 32, 24, (2, 3, 7), True),
]
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]
_POSITION_DTYPES = [infinicore.int32, infinicore.int64]
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
}


def make_positions(num_tokens):
    return torch.stack(
        [
            torch.arange(num_tokens, dtype=torch.int64) * 2,
            torch.arange(num_tokens, dtype=torch.int64) * 2 + 1,
            torch.arange(num_tokens, dtype=torch.int64) * 2 + 2,
        ]
    )


def parse_test_cases():
    test_cases = []
    for (
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_size,
        rotary_dim,
        sections,
        interleaved,
    ) in _TEST_CASES_DATA:
        max_positions = num_tokens * 2 + 4
        positions = make_positions(num_tokens)
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP[dtype]
            for position_dtype in _POSITION_DTYPES:
                q_spec = TensorSpec.from_tensor(
                    (num_tokens, num_q_heads * head_size), None, dtype
                )
                k_spec = TensorSpec.from_tensor(
                    (num_tokens, num_kv_heads * head_size), None, dtype
                )
                cos_spec = TensorSpec.from_tensor(
                    (max_positions, rotary_dim // 2), None, dtype
                )
                sin_spec = TensorSpec.from_tensor(
                    (max_positions, rotary_dim // 2), None, dtype
                )
                pos_spec = TensorSpec.from_tensor(
                    positions.shape,
                    None,
                    position_dtype,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=positions,
                )
                kwargs = {
                    "head_size": head_size,
                    "rotary_dim": rotary_dim,
                    "section_t": sections[0],
                    "section_h": sections[1],
                    "section_w": sections[2],
                    "interleaved": interleaved,
                }
                test_cases.append(
                    TestCase(
                        inputs=[q_spec, k_spec, cos_spec, sin_spec, pos_spec],
                        kwargs=kwargs,
                        comparison_target=None,
                        tolerance=tolerance,
                        output_count=2,
                        description="MRoPE - OUT_OF_PLACE",
                    )
                )
                test_cases.append(
                    TestCase(
                        inputs=[q_spec, k_spec, cos_spec, sin_spec, pos_spec],
                        kwargs=kwargs,
                        output_specs=[
                            TensorSpec.from_tensor(
                                (num_tokens, num_q_heads * head_size), None, dtype
                            ),
                            TensorSpec.from_tensor(
                                (num_tokens, num_kv_heads * head_size), None, dtype
                            ),
                        ],
                        comparison_target="out",
                        tolerance=tolerance,
                        output_count=2,
                        description="MRoPE - INPLACE(out)",
                    )
                )
    return test_cases


def axis_for_dim(dim, section_t, section_h, section_w, interleaved):
    if interleaved:
        mod = dim % 3
        if mod == 1 and dim < section_h * 3:
            return 1
        if mod == 2 and dim < section_w * 3:
            return 2
        return 0
    if dim < section_t:
        return 0
    if dim < section_t + section_h:
        return 1
    return 2


def torch_mrope_one(
    x,
    cos,
    sin,
    positions,
    head_size,
    rotary_dim,
    section_t,
    section_h,
    section_w,
    interleaved,
):
    num_tokens = x.shape[0]
    num_heads = x.shape[1] // head_size
    half = rotary_dim // 2
    x = x.reshape(num_tokens, num_heads, head_size)
    out = x.clone()
    cos_row = torch.empty((num_tokens, half), dtype=torch.float32, device=x.device)
    sin_row = torch.empty((num_tokens, half), dtype=torch.float32, device=x.device)
    has_axes = positions.ndim == 2
    for i in range(half):
        axis = axis_for_dim(i, section_t, section_h, section_w, interleaved)
        pos = positions[axis] if has_axes else positions
        cos_row[:, i] = cos[pos, i].float()
        sin_row[:, i] = sin[pos, i].float()
    x0 = x[:, :, :half].float()
    x1 = x[:, :, half:rotary_dim].float()
    cos_row = cos_row[:, None, :]
    sin_row = sin_row[:, None, :]
    out[:, :, :half] = (x0 * cos_row - x1 * sin_row).to(out.dtype)
    out[:, :, half:rotary_dim] = (x1 * cos_row + x0 * sin_row).to(out.dtype)
    return out.reshape(num_tokens, num_heads * head_size)


def torch_mrope(q, k, cos, sin, positions, **kwargs):
    q_out = torch_mrope_one(q, cos, sin, positions, **kwargs)
    k_out = torch_mrope_one(k, cos, sin, positions, **kwargs)
    return q_out, k_out


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("MRoPE")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, out=None, **kwargs):
        q_out, k_out = torch_mrope(*args, **kwargs)
        if out is not None:
            out[0].copy_(q_out)
            out[1].copy_(k_out)
            return out
        return q_out, k_out

    def infinicore_operator(self, q, k, cos, sin, positions, out=None, **kwargs):
        return infinicore.mrope(q, k, cos, sin, positions, out=out, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
