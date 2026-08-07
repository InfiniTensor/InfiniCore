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

# Test cases:
# (qkv_shape, qkv_strides, state_shape, weight_shape, bias_shape,
#  cu_seqlens, initial_state_indices, final_state_indices)
_TEST_CASES_DATA = [
    ((2, 5, 4), None, (2, 4, 3), (4, 1, 4), None, None, None, None),
    ((2, 4, 12), (72, 18, 1), (2, 12, 3), (12, 1, 4), (12,), None, None, None),
    ((1, 7, 8), None, (2, 8, 3), (8, 1, 4), (8,), (0, 3, 7), None, None),
    ((1, 7, 10), None, (4, 10, 3), (10, 1, 4), (10,), (0, 2, 7), (1, 3), (0, 2)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 2e-2, "rtol": 2e-2},
    infinicore.bfloat16: {"atol": 5e-2, "rtol": 5e-2},
    infinicore.float32: {"atol": 1e-4, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def torch_causal_conv1d_ref(
    qkv,
    conv_state,
    weight,
    bias=None,
    cu_seqlens=None,
    initial_state_indices=None,
    final_state_indices=None,
):
    # qkv/out: [B, T, C] in padded mode, or [1, total_tokens, C] with cu_seqlens.
    # conv_state: [B/num_requests, C, state_len] or [pool_size, C, state_len].
    out = torch.empty_like(qkv)
    state_len = conv_state.shape[2]

    if cu_seqlens is None:
        cu = None
        request_count = qkv.shape[0]
    else:
        cu = cu_seqlens.cpu().tolist()
        request_count = len(cu) - 1

    for req in range(request_count):
        if cu is None:
            batch = req
            start = 0
            end = qkv.shape[1]
        else:
            batch = 0
            start = cu[req]
            end = cu[req + 1]

        read_slot = req
        if initial_state_indices is not None:
            read_slot = int(initial_state_indices[req].item())

        for c in range(qkv.shape[2]):
            history = torch.cat(
                [
                    conv_state[read_slot, c].float(),
                    qkv[batch, start:end, c].float(),
                ],
                dim=0,
            )
            filt = weight[c, 0].float()
            values = []
            for t in range(end - start):
                value = (history[t : t + state_len + 1] * filt).sum()
                if bias is not None:
                    value = value + bias[c].float()
                values.append(value)
            out[batch, start:end, c] = torch.stack(values).to(out.dtype)

            if final_state_indices is not None:
                write_slot = int(final_state_indices[req].item())
                conv_state[write_slot, c].copy_(
                    history[end - start : end - start + state_len].to(conv_state.dtype)
                )

    return out


def _manual_i32(values):
    return TensorSpec.from_tensor(
        (len(values),),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor(values, dtype=torch.int32),
    )


def parse_test_cases():
    tests = []
    for (
        qkv_shape,
        qkv_strides,
        state_shape,
        weight_shape,
        bias_shape,
        cu_seqlens,
        initial_state_indices,
        final_state_indices,
    ) in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            inputs = [
                TensorSpec.from_tensor(qkv_shape, qkv_strides, dtype),
                TensorSpec.from_tensor(state_shape, None, dtype),
                TensorSpec.from_tensor(weight_shape, None, dtype),
            ]

            if bias_shape is not None:
                inputs.append(TensorSpec.from_tensor(bias_shape, None, dtype))

            if cu_seqlens is not None:
                inputs.append(_manual_i32(cu_seqlens))

            if initial_state_indices is not None:
                inputs.append(_manual_i32(initial_state_indices))
                inputs.append(_manual_i32(final_state_indices))

            tests.append(
                TestCase(
                    inputs=inputs,
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="CausalConv1d - OUT_OF_PLACE",
                )
            )

            if final_state_indices is not None:
                tests.append(
                    TestCase(
                        inputs=inputs,
                        kwargs={},
                        output_spec=None,
                        comparison_target=1,
                        tolerance=_TOLERANCE_MAP[dtype],
                        description="CausalConv1d - UPDATED_STATE",
                    )
                )

    # Large indexed state update: C=2048 keeps all eight Ascend blocks active
    # for FP16/BF16 under the 512-byte alignment rule and catches cache-line
    # races that small shapes cannot expose.
    for dtype in _TENSOR_DTYPES:
        inputs = [
            TensorSpec.from_tensor((1, 1, 2048), None, dtype),
            TensorSpec.from_tensor((2, 2048, 3), None, dtype),
            TensorSpec.from_tensor((2048, 1, 4), None, dtype),
            TensorSpec.from_tensor((2048,), None, dtype),
            _manual_i32((0, 1)),
            _manual_i32((1,)),
            _manual_i32((1,)),
        ]
        tests.append(
            TestCase(
                inputs=inputs,
                kwargs={},
                output_spec=None,
                comparison_target=1,
                tolerance=_TOLERANCE_MAP[dtype],
                description="CausalConv1d - MULTICORE_UPDATED_STATE",
            )
        )

    return tests


def _unpack_args(args):
    bias = None
    cu_seqlens = None
    initial_state_indices = None
    final_state_indices = None

    if len(args) >= 4:
        bias = args[3]
    if len(args) >= 5:
        cu_seqlens = args[4]
    if len(args) >= 7:
        initial_state_indices = args[5]
        final_state_indices = args[6]

    return bias, cu_seqlens, initial_state_indices, final_state_indices


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("CausalConv1d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, qkv, conv_state, weight, *args, **kwargs):
        bias, cu_seqlens, initial_state_indices, final_state_indices = _unpack_args(
            (qkv, conv_state, weight, *args)
        )
        return torch_causal_conv1d_ref(
            qkv,
            conv_state,
            weight,
            bias=bias,
            cu_seqlens=cu_seqlens,
            initial_state_indices=initial_state_indices,
            final_state_indices=final_state_indices,
        )

    def infinicore_operator(self, qkv, conv_state, weight, *args, **kwargs):
        bias, cu_seqlens, initial_state_indices, final_state_indices = _unpack_args(
            (qkv, conv_state, weight, *args)
        )
        return infinicore.nn.functional.causal_conv1d(
            qkv,
            conv_state,
            weight,
            bias,
            cu_seqlens=cu_seqlens,
            initial_state_indices=initial_state_indices,
            final_state_indices=final_state_indices,
        )


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
