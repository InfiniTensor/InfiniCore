"""Compare packed Mamba-2 outputs and state against an independent recurrence."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
)
from framework import (
    TestCase as OperatorTestCase,
)

import infinicore


def torch_mamba2_scan(x, dt, b, c, a, d, dt_bias, state, offsets, initial, final):
    """Use token-by-token FP32 recurrence, independent of the chunk algorithm."""
    output = torch.empty_like(x)
    boundaries = offsets.cpu().tolist()
    source_rows = initial.cpu().tolist()
    target_rows = final.cpu().tolist()
    heads_per_group = x.shape[1] // b.shape[1]
    source = state.clone()
    for request, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        current = source[source_rows[request]].clone()
        for token in range(start, end):
            step = torch.nn.functional.softplus(dt[token].float() + dt_bias)
            decay = torch.exp(step * a)
            bt = b[token].float().repeat_interleave(heads_per_group, dim=0)
            ct = c[token].float().repeat_interleave(heads_per_group, dim=0)
            xt = x[token].float()
            current = (
                decay[:, None, None] * current
                + step[:, None, None] * xt[:, :, None] * bt[:, None, :]
            )
            output[token] = (current * ct[:, None, :]).sum(-1) + d[:, None] * xt
        state[target_rows[request]] = current
    return output, state


def _spec(tensor, dtype):
    return TensorSpec.from_tensor(
        tuple(tensor.shape),
        None,
        dtype,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=tensor,
    )


def parse_test_cases():
    cases = []
    shapes = [
        ([length], 4, 7, 2, 9)
        for length in (1, 2, 3, 4, 5, 255, 256, 257, 511, 512, 513)
    ]
    shapes += [([1, 3, 257], 4, 7, 1, 33), ([257, 2, 256], 4, 7, 2, 128)]
    shapes += [([1], 24, 64, 1, 128), ([257], 24, 64, 1, 128)]
    shapes += [([1, 1, 1, 1], 24, 64, 1, 128), ([5], 4, 7, 2, 256)]
    shapes += [([1025], 4, 7, 2, 33)]
    generator = torch.Generator().manual_seed(20260916)
    for lengths, heads, head_dim, groups, state_size in shapes:
        tokens, requests = sum(lengths), len(lengths)
        pool = 2 * requests + 2
        offsets = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int()
        for torch_dtype, infini_dtype, tolerance in (
            (torch.float32, infinicore.float32, 1e-4),
            (torch.float16, infinicore.float16, 3e-3),
            (torch.bfloat16, infinicore.bfloat16, 2e-2),
        ):

            def random(shape):
                return (torch.randn(shape, generator=generator) * 0.2).to(torch_dtype)

            state = (
                torch.randn(pool, heads, head_dim, state_size, generator=generator)
                * 0.1
            )
            state[0].zero_()
            initial = torch.arange(1, requests + 1, dtype=torch.int32)
            initial[0] = 0
            # Nonzero requests update their own slots; the first uses a distinct slot.
            final = torch.arange(1, requests + 1, dtype=torch.int32)
            final[0] = pool - 1
            tensors = [
                _spec(random((tokens, heads, head_dim)), infini_dtype),
                _spec(random((tokens, heads)), infini_dtype),
                _spec(random((tokens, groups, state_size)), infini_dtype),
                _spec(random((tokens, groups, state_size)), infini_dtype),
                _spec(
                    -torch.arange(1, heads + 1, dtype=torch.float32), infinicore.float32
                ),
                _spec(torch.ones(heads), infinicore.float32),
                _spec(
                    torch.linspace(-80, 80, heads)
                    if lengths == [1025]
                    else torch.full((heads,), -3.0),
                    infinicore.float32,
                ),
                _spec(state, infinicore.float32),
                _spec(offsets, infinicore.int32),
                _spec(initial, infinicore.int32),
                _spec(final, infinicore.int32),
            ]
            cases.append(
                OperatorTestCase(
                    inputs=tensors,
                    kwargs={},
                    output_spec=None,
                    comparison_target=[7],
                    tolerance={"atol": tolerance, "rtol": tolerance},
                    description=f"Mamba2Scan lengths={lengths}, groups={groups}: output and entire state pool",
                    output_count=2,
                )
            )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Mamba2Scan")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_mamba2_scan(*args)

    def infinicore_operator(self, *args, **kwargs):
        output = infinicore.nn.functional.mamba2_scan(*args)
        return output, args[7]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA device is required.")
@pytest.mark.parametrize(
    "invalid",
    [
        "activation_dtype",
        "parameter_dtype",
        "state_dtype",
        "offset_dtype",
        "index_count",
        "bc_shape",
        "zero_only_pool",
        "noncontiguous",
        "head_groups",
    ],
)
def test_invalid_descriptor_is_rejected(invalid):
    args = [
        torch.zeros(4, 4, 8, device="cuda"),
        torch.zeros(4, 4, device="cuda"),
        torch.zeros(4, 2, 16, device="cuda"),
        torch.zeros(4, 2, 16, device="cuda"),
        -torch.ones(4, device="cuda"),
        torch.ones(4, device="cuda"),
        torch.zeros(4, device="cuda"),
        torch.zeros(4, 4, 8, 16, device="cuda"),
        torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
    ]
    if invalid == "activation_dtype":
        args[1] = args[1].half()
    elif invalid == "parameter_dtype":
        args[4] = args[4].half()
    elif invalid == "state_dtype":
        args[7] = args[7].half()
    elif invalid == "offset_dtype":
        args[8] = args[8].long()
    elif invalid == "index_count":
        args[9] = args[9].repeat(2)
    elif invalid == "bc_shape":
        args[3] = args[3][:, :1].contiguous()
    elif invalid == "zero_only_pool":
        args[7] = args[7][:1]
    elif invalid == "noncontiguous":
        args[0] = args[0].transpose(0, 1)
    elif invalid == "head_groups":
        args[2] = args[3] = torch.zeros(4, 3, 16, device="cuda")
    torch.cuda.synchronize()
    wrapped = [
        infinicore.strided_from_blob(
            value.data_ptr(),
            list(value.shape),
            list(value.stride()),
            dtype=infinicore.utils.to_infinicore_dtype(value.dtype),
            device=infinicore.device("cuda", value.device.index),
        )
        for value in args
    ]
    with pytest.raises(RuntimeError):
        infinicore.nn.functional.mamba2_scan(*wrapped)


if __name__ == "__main__":
    GenericTestRunner(OpTest).run_and_exit()
