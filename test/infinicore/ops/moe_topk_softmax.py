import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
import torch.nn.functional as F
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)
from infinicore.lib import _infinicore


# Test case format:
# (shape, topk, renormalize, moe_softcapping, bias_mode, description)
_TEST_CASES_DATA = [
    ((2, 4), 2, True, 0.0, "none", "ordinary softmax top-k without correction bias"),
    (
        (2, 4),
        2,
        True,
        0.0,
        "zero",
        "zero correction bias matches ordinary softmax top-k",
    ),
    (
        (3, 4),
        2,
        True,
        0.0,
        "force_last",
        "bias changes selected indices, weights use original probs",
    ),
    (
        (2, 6),
        2,
        True,
        0.0,
        "force_middle",
        "bias semantics on non-power-of-two expert count",
    ),
    (
        (4, 64),
        6,
        True,
        0.0,
        "small_random",
        "power-of-two expert count with random correction bias",
    ),
    (
        (2, 8),
        3,
        False,
        1.5,
        "force_last",
        "softcapping with unnormalized selected weights",
    ),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 2e-3, "rtol": 2e-3},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def _make_logits(shape, seed, forced_expert=None):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(shape, generator=generator, dtype=torch.float32) * 1.25
    if forced_expert is not None:
        logits[:, forced_expert] -= 4.0
    return logits


def _make_correction_bias(logits, mode, seed):
    num_experts = logits.shape[-1]
    if mode == "none":
        return None
    if mode == "zero":
        return torch.zeros(num_experts, dtype=torch.float32)
    if mode == "small_random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(num_experts, generator=generator, dtype=torch.float32) * 0.05

    if mode == "force_last":
        target = num_experts - 1
    elif mode == "force_middle":
        target = num_experts // 2
    else:
        raise ValueError(f"Unsupported bias mode: {mode}")

    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    other_probs = torch.cat([probs[:, :target], probs[:, target + 1 :]], dim=-1)
    required_margin = (other_probs.max(dim=-1).values - probs[:, target]).max().item()
    bias = torch.zeros(num_experts, dtype=torch.float32)
    bias[target] = required_margin + 0.25
    return bias


def torch_moe_topk_softmax(
    gating_output, topk, correction_bias=None, renormalize=True, moe_softcapping=0.0
):
    logits = gating_output.to(torch.float32)
    if moe_softcapping != 0.0:
        logits = torch.tanh(logits / moe_softcapping) * moe_softcapping

    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    choice_scores = probs
    if correction_bias is not None:
        choice_scores = choice_scores + correction_bias.reshape(1, -1).to(torch.float32)

    _, selected_experts = torch.topk(choice_scores, topk, dim=-1)
    routing_weights = torch.gather(probs, dim=-1, index=selected_experts)
    if renormalize:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    return routing_weights, selected_experts.to(torch.int32)


def parse_test_cases():
    test_cases = []
    for case_idx, (
        shape,
        topk,
        renormalize,
        softcap,
        bias_mode,
        description,
    ) in enumerate(_TEST_CASES_DATA):
        num_experts = shape[-1]
        forced_expert = None
        if bias_mode == "force_last":
            forced_expert = num_experts - 1
        elif bias_mode == "force_middle":
            forced_expert = num_experts // 2

        logits = _make_logits(shape, seed=2027 + case_idx, forced_expert=forced_expert)
        bias = _make_correction_bias(logits, bias_mode, seed=4099 + case_idx)

        for dtype in _TENSOR_DTYPES:
            input_spec = TensorSpec.from_tensor(
                shape,
                None,
                dtype,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=logits,
                name="gating_output",
            )
            inputs = [input_spec]
            if bias is not None:
                inputs.append(
                    TensorSpec.from_tensor(
                        (num_experts,),
                        None,
                        infinicore.float32,
                        init_mode=TensorInitializer.MANUAL,
                        set_tensor=bias,
                        name="correction_bias",
                    )
                )

            kwargs = {
                "topk": topk,
                "renormalize": renormalize,
                "moe_softcapping": softcap,
            }
            case_name = (
                f"moe_topk_softmax - {description} - dtype={dtype}, "
                f"shape={shape}, topk={topk}, bias={bias_mode}"
            )

            test_cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE_MAP[dtype],
                    description=f"{case_name} - OUT_OF_PLACE",
                    output_count=2,
                )
            )

            out_shape = (shape[0], topk)
            test_cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs=kwargs.copy(),
                    output_specs=[
                        TensorSpec.from_tensor(
                            out_shape, None, infinicore.float32, name="topk_weights"
                        ),
                        TensorSpec.from_tensor(
                            out_shape, None, infinicore.int32, name="topk_indices"
                        ),
                    ],
                    comparison_target="out",
                    tolerance=_TOLERANCE_MAP[dtype],
                    description=f"{case_name} - INPLACE(out)",
                    output_count=2,
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("moe_topk_softmax")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self,
        gating_output,
        correction_bias=None,
        topk=2,
        renormalize=True,
        moe_softcapping=0.0,
        out=None,
        **kwargs,
    ):
        values, indices = torch_moe_topk_softmax(
            gating_output,
            topk,
            correction_bias=correction_bias,
            renormalize=renormalize,
            moe_softcapping=moe_softcapping,
        )
        if out is not None:
            out_v, out_i = out
            out_v.copy_(values)
            out_i.copy_(indices)
        return values, indices

    def infinicore_operator(
        self,
        gating_output,
        correction_bias=None,
        topk=2,
        renormalize=True,
        moe_softcapping=0.0,
        out=None,
        **kwargs,
    ):
        if out is None:
            values = infinicore.empty(
                (gating_output.shape[0], topk),
                dtype=infinicore.float32,
                device=gating_output.device,
            )
            indices = infinicore.empty(
                (gating_output.shape[0], topk),
                dtype=infinicore.int32,
                device=gating_output.device,
            )
        else:
            values, indices = out

        _infinicore.moe_topk_softmax_(
            values._underlying,
            indices._underlying,
            gating_output._underlying,
            correction_bias._underlying if correction_bias is not None else None,
            renormalize,
            moe_softcapping,
        )
        return values, indices


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
