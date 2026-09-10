import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)

import infinicore

_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]
_TOLERANCE = {
    infinicore.float16: {"atol": 7e-2, "rtol": 7e-2},
    infinicore.bfloat16: {"atol": 1.5e-1, "rtol": 1.5e-1},
    infinicore.float32: {"atol": 8e-4, "rtol": 8e-4},
}


def dequantize_mxfp4(packed, scales):
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    values = magnitudes[(codes & 0x07).to(torch.int64)]
    values = torch.where((codes & 0x08) != 0, -values, values)
    exponents = scales.to(torch.int32).sub(127).repeat_interleave(32, dim=-1)
    return torch.ldexp(values, exponents)


def situ(gate, up):
    beta = 4.0
    linear_beta = 25.0
    situ_gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    bounded_up = linear_beta * torch.tanh(up / linear_beta)
    return situ_gate * bounded_up


def torch_fused_moe_mxfp4(
    input,
    selected_experts,
    routing_weights,
    w13_packed,
    w13_scale,
    w2_packed,
    w2_scale,
    activation,
):
    w13 = dequantize_mxfp4(w13_packed, w13_scale)
    w2 = dequantize_mxfp4(w2_packed, w2_scale)
    output = torch.zeros_like(input, dtype=torch.float32)
    for token in range(input.shape[0]):
        for route in range(selected_experts.shape[1]):
            expert = int(selected_experts[token, route])
            if expert < 0 or expert >= w13.shape[0]:
                continue
            gate_up = F.linear(input[token].float(), w13[expert])
            gate, up = gate_up.chunk(2, dim=-1)
            activated = situ(gate, up) if activation == 2 else F.silu(gate) * up
            activated = activated.to(input.dtype).float()
            output[token] += (
                F.linear(activated, w2[expert]) * routing_weights[token, route]
            )
    return output.to(input.dtype)


def make_cases():
    generator = torch.Generator(device="cpu").manual_seed(20260731)
    configs = [
        (1, 64, 64, 8, 3, 2, "decode SiTU"),
        (5, 64, 96, 6, 2, 1, "prefill SwiGLU"),
        (7, 128, 64, 5, 2, 2, "prefill SiTU"),
    ]
    cases = []
    for (
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        activation,
        description,
    ) in configs:
        input_data = torch.randn((num_tokens, hidden_size), generator=generator) * 0.2
        ids = torch.randint(
            0,
            num_experts,
            (num_tokens, topk),
            generator=generator,
            dtype=torch.int32,
        )
        if num_tokens > 1:
            ids[-1, -1] = -1
        raw_routes = torch.rand((num_tokens, topk), generator=generator)
        routing = raw_routes / raw_routes.sum(dim=-1, keepdim=True)
        w13_packed = torch.randint(
            0,
            256,
            (num_experts, 2 * intermediate_size, hidden_size // 2),
            generator=generator,
            dtype=torch.uint8,
        )
        w13_scale = torch.randint(
            123,
            129,
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            generator=generator,
            dtype=torch.uint8,
        )
        w2_packed = torch.randint(
            0,
            256,
            (num_experts, hidden_size, intermediate_size // 2),
            generator=generator,
            dtype=torch.uint8,
        )
        w2_scale = torch.randint(
            123,
            129,
            (num_experts, hidden_size, intermediate_size // 32),
            generator=generator,
            dtype=torch.uint8,
        )
        for dtype in _DTYPES:
            tensors = [
                (input_data, dtype, "input"),
                (ids, infinicore.int32, "selected_experts"),
                (routing, infinicore.float32, "routing_weights"),
                (w13_packed, infinicore.uint8, "w13_packed"),
                (w13_scale, infinicore.uint8, "w13_scale"),
                (w2_packed, infinicore.uint8, "w2_packed"),
                (w2_scale, infinicore.uint8, "w2_scale"),
            ]
            inputs = [
                TensorSpec.from_tensor(
                    tuple(tensor.shape),
                    None,
                    tensor_dtype,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=tensor,
                    name=name,
                )
                for tensor, tensor_dtype, name in tensors
            ]
            cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs={"activation": activation},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE[dtype],
                    description=f"fused_moe_mxfp4 - {description} - dtype={dtype}",
                )
            )

    # Kimi-K3 reaches this launch size during long prefills. CUDA-compatible
    # backends limit a launch to a 32-bit total-thread range, so the W13 grid
    # must be split once route_count * intermediate_size exceeds that limit.
    num_tokens = 293
    hidden_size = 64
    intermediate_size = 3584
    num_experts = 1
    topk = 16
    input_data = torch.randn((num_tokens, hidden_size), generator=generator) * 0.2
    ids = torch.full((num_tokens, topk), -1, dtype=torch.int32)
    ids[-1, -1] = 0
    routing = torch.full((num_tokens, topk), 1.0 / topk)
    tensors = [
        (input_data, infinicore.bfloat16, "input"),
        (ids, infinicore.int32, "selected_experts"),
        (routing, infinicore.float32, "routing_weights"),
        (
            torch.full(
                (num_experts, 2 * intermediate_size, hidden_size // 2),
                0x22,
                dtype=torch.uint8,
            ),
            infinicore.uint8,
            "w13_packed",
        ),
        (
            torch.full(
                (num_experts, 2 * intermediate_size, hidden_size // 32),
                127,
                dtype=torch.uint8,
            ),
            infinicore.uint8,
            "w13_scale",
        ),
        (
            torch.full(
                (num_experts, hidden_size, intermediate_size // 2),
                0x22,
                dtype=torch.uint8,
            ),
            infinicore.uint8,
            "w2_packed",
        ),
        (
            torch.full(
                (num_experts, hidden_size, intermediate_size // 32),
                127,
                dtype=torch.uint8,
            ),
            infinicore.uint8,
            "w2_scale",
        ),
    ]
    cases.append(
        TestCase(
            inputs=[
                TensorSpec.from_tensor(
                    tuple(tensor.shape),
                    None,
                    tensor_dtype,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=tensor,
                    name=name,
                )
                for tensor, tensor_dtype, name in tensors
            ],
            kwargs={"activation": 2},
            output_spec=None,
            comparison_target=None,
            tolerance=_TOLERANCE[infinicore.bfloat16],
            description="fused_moe_mxfp4 - large-grid regression",
        )
    )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("fused_moe_mxfp4")

    def get_test_cases(self):
        return make_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_fused_moe_mxfp4(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.nn.functional.fused_moe_mxfp4(*args, **kwargs)


if __name__ == "__main__":
    GenericTestRunner(OpTest).run_and_exit()
