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
    infinicore.float16: {"atol": 2e-1, "rtol": 2e-1},
    infinicore.bfloat16: {"atol": 3e-1, "rtol": 3e-1},
    infinicore.float32: {"atol": 2e-4, "rtol": 2e-4},
}


def unpack_int4(packed):
    values = packed.to(torch.uint8)
    unpacked = torch.stack((values >> 4, values & 0x0F), dim=-1).flatten(-2)
    unpacked = unpacked.to(torch.int8)
    return torch.where(unpacked >= 8, unpacked - 16, unpacked)


def shuffle_aiter_weight(packed_weight):
    n, packed_k = packed_weight.shape[-2:]
    values = packed_weight.to(torch.uint8)
    unpacked = torch.stack((values >> 4, values & 0x0F), dim=-1).reshape(
        *values.shape[:-2], n, -1
    )
    blocks = unpacked.reshape(*unpacked.shape[:-1], -1, 8)
    blocked = ((blocks[..., :4] << 4) | blocks[..., 4:]).reshape(
        *values.shape[:-2], n, packed_k
    )
    n_tile = 256 if n % 256 == 0 else n
    shuffled = blocked.transpose(-2, -1).reshape(
        *values.shape[:-2], packed_k // 32, 32, n // n_tile, n_tile
    )
    shuffled = shuffled.permute(
        *range(shuffled.ndim - 4), -4, -2, -1, -3
    ).contiguous()
    shuffled = shuffled.reshape(
        *values.shape[:-2], packed_k // 32, n // n_tile, n_tile // 32, 32, 4, 8
    )
    return shuffled.transpose(-4, -3).contiguous().reshape_as(packed_weight).to(torch.int8)


def unshuffle_aiter_weight(shuffled_weight):
    n, packed_k = shuffled_weight.shape[-2:]
    n_tile = 256 if n % 256 == 0 else n
    prefix = shuffled_weight.shape[:-2]
    shuffled = shuffled_weight.to(torch.uint8).reshape(
        *prefix, packed_k // 32, n // n_tile, n_tile // 32, 4, 32, 8
    )
    shuffled = shuffled.transpose(-4, -3).contiguous().reshape(
        *prefix, packed_k // 32, n // n_tile, n_tile, 32
    )
    blocked = shuffled.permute(
        *range(shuffled.ndim - 4), -4, -1, -3, -2
    ).contiguous().reshape(*prefix, packed_k, n).transpose(-2, -1)
    pairs = torch.stack((blocked >> 4, blocked & 0x0F), dim=-1).reshape(
        *prefix, n, -1, 8
    )
    unpacked = pairs[..., [0, 2, 4, 6, 1, 3, 5, 7]]
    return ((unpacked[..., ::2] << 4) | unpacked[..., 1::2]).reshape(
        *prefix, n, packed_k
    ).to(torch.int8)


def quantize_per_token(input):
    rows = input.float().reshape(-1, input.shape[-1])
    scale = rows.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    quantized = torch.round(rows / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def situ(gate, up):
    return 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * (
        25.0 * torch.tanh(up / 25.0)
    )


def torch_fused_moe_w4a8(
    input,
    selected_experts,
    routing_weights,
    w13_packed,
    w13_scale,
    w2_packed,
    w2_scale,
    activation,
    weights_are_aiter_shuffled=False,
):
    if weights_are_aiter_shuffled:
        w13_packed = unshuffle_aiter_weight(w13_packed)
        w2_packed = unshuffle_aiter_weight(w2_packed)
        w13_scale = w13_scale * 16.0
        w2_scale = w2_scale * 16.0
    input_q, input_scale = quantize_per_token(input)
    w13 = unpack_int4(w13_packed).float()
    w2 = unpack_int4(w2_packed).float()
    output = torch.zeros_like(input, dtype=torch.float32)
    for token in range(input.shape[0]):
        for route in range(selected_experts.shape[1]):
            expert = int(selected_experts[token, route])
            if expert < 0 or expert >= w13.shape[0]:
                continue
            gate_up = torch.matmul(input_q[token].float(), w13[expert].t())
            gate_up *= input_scale[token] * w13_scale[expert, :, 0]
            gate, up = gate_up.chunk(2, dim=-1)
            activated = situ(gate, up) if activation == 2 else F.silu(gate) * up
            activated = activated.to(input.dtype)
            activated_q, activated_scale = quantize_per_token(activated)
            down = torch.matmul(activated_q[0].float(), w2[expert].t())
            down *= activated_scale[0] * w2_scale[expert, :, 0]
            output[token] += down * routing_weights[token, route]
    return output.to(input.dtype)


def make_cases():
    generator = torch.Generator(device="cpu").manual_seed(20260911)
    configs = [
        (1, 64, 64, 8, 3, 2, "decode SiTU"),
        (1, 64, 64, 896, 16, 2, "Kimi decode route packing"),
        (5, 64, 96, 6, 2, 1, "prefill SwiGLU"),
    ]
    cases = []
    for tokens, hidden, intermediate, experts, topk, activation, description in configs:
        input_data = torch.randn((tokens, hidden), generator=generator) * 0.2
        ids = torch.randint(
            0, experts, (tokens, topk), generator=generator, dtype=torch.int32
        )
        if tokens > 1:
            ids[-1, -1] = -1
        raw_routes = torch.rand((tokens, topk), generator=generator)
        routing = raw_routes / raw_routes.sum(dim=-1, keepdim=True)
        w13_packed = torch.randint(
            -128,
            128,
            (experts, 2 * intermediate, hidden // 2),
            generator=generator,
            dtype=torch.int8,
        )
        w13_scale = torch.rand(
            (experts, 2 * intermediate, 1), generator=generator
        ) * 0.02
        w2_packed = torch.randint(
            -128,
            128,
            (experts, hidden, intermediate // 2),
            generator=generator,
            dtype=torch.int8,
        )
        w2_scale = torch.rand((experts, hidden, 1), generator=generator) * 0.02
        for dtype in _DTYPES:
            tensors = [
                (input_data, dtype, "input"),
                (ids, infinicore.int32, "selected_experts"),
                (routing, infinicore.float32, "routing_weights"),
                (w13_packed, infinicore.int8, "w13_packed"),
                (w13_scale, infinicore.float32, "w13_scale"),
                (w2_packed, infinicore.int8, "w2_packed"),
                (w2_scale, infinicore.float32, "w2_scale"),
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
                    description=f"fused_moe_w4a8 - {description} - dtype={dtype}",
                )
            )

        if hidden % 64 != 0 or intermediate % 64 != 0:
            continue
        for optimized_dtype in (infinicore.float16, infinicore.bfloat16):
            optimized_tensors = [
                (input_data, optimized_dtype, "input"),
                (ids, infinicore.int32, "selected_experts"),
                (routing, infinicore.float32, "routing_weights"),
                (shuffle_aiter_weight(w13_packed), infinicore.int8, "w13_packed"),
                (w13_scale / 16.0, infinicore.float32, "w13_scale"),
                (shuffle_aiter_weight(w2_packed), infinicore.int8, "w2_packed"),
                (w2_scale / 16.0, infinicore.float32, "w2_scale"),
            ]
            optimized_inputs = [
                TensorSpec.from_tensor(
                    tuple(tensor.shape),
                    None,
                    tensor_dtype,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=tensor,
                    name=name,
                )
                for tensor, tensor_dtype, name in optimized_tensors
            ]
            cases.append(
                TestCase(
                    inputs=optimized_inputs,
                    kwargs={
                        "activation": activation,
                        "weights_are_aiter_shuffled": True,
                    },
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE[optimized_dtype],
                    description=(
                        f"fused_moe_w4a8 - AITER {description} "
                        f"- dtype={optimized_dtype}"
                    ),
                )
            )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("fused_moe_w4a8")

    def get_test_cases(self):
        return make_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_fused_moe_w4a8(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.nn.functional.fused_moe_w4a8(*args, **kwargs)


if __name__ == "__main__":
    GenericTestRunner(OpTest).run_and_exit()
