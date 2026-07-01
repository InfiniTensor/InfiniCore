import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as torch_F
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)

import infinicore

# Test cases:
# (B, T, Hk, Hv, Dk, Dv, use_qk_l2norm, strided_qkv)
_TEST_CASES = [
    (7, 1, 40, 40, 128, 128, True, False),
    (5, 1, 64, 64, 128, 128, False, False),
    (1, 1, 8, 8, 64, 64, True, False),
    (2, 1, 4, 8, 64, 64, False, False),
    (2, 1, 4, 8, 64, 64, True, True),
]
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 5e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
}


def ref_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    use_qk_l2norm=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = torch_F.normalize(query, p=2, dim=-1)
        key = torch_F.normalize(key, p=2, dim=-1)

    query, key, value, beta, g = [
        x.contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    state = initial_state.transpose(-1, -2).contiguous().to(torch.float32).clone()
    batch_size, sequence_length, key_heads, _ = key.shape
    value_heads, v_head_dim = value.shape[2], value.shape[-1]
    value_heads_per_key_head = value_heads // key_heads
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    out = torch.zeros(
        batch_size,
        sequence_length,
        value_heads,
        v_head_dim,
        device=value.device,
        dtype=torch.float32,
    )

    for i in range(sequence_length):
        for vh in range(value_heads):
            kh = vh // value_heads_per_key_head
            q_t = query[:, i, kh]
            k_t = key[:, i, kh]
            v_t = value[:, i, vh]
            g_t = g[:, i, vh].exp().view(batch_size, 1, 1)
            beta_t = beta[:, i, vh].view(batch_size, 1)
            state_t = state[:, vh]

            state_t = state_t * g_t
            kv_mem = (state_t * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state_t = state_t + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            state[:, vh] = state_t
            out[:, i, vh] = (state_t * q_t.unsqueeze(-1)).sum(dim=-2)

    return out.contiguous().to(initial_dtype), state.transpose(-1, -2).contiguous().to(
        initial_dtype
    )


def strided_bthd_strides(shape):
    _, T, H, D = shape
    return (8 * T * H * D, 4 * H * D, 2 * D, 1)


def tensor_spec(shape, dtype, strides=None):
    return TensorSpec.from_tensor(shape, strides, dtype, scale=0.2, bias=-0.1)


def gate_spec(shape):
    return TensorSpec.from_tensor(
        shape, None, infinicore.float32, scale=0.02, bias=-0.01
    )


def beta_spec(shape):
    return TensorSpec.from_tensor(shape, None, infinicore.float32, scale=0.5, bias=0.0)


def index_spec(values):
    values = torch.tensor(values, dtype=torch.int64)
    return TensorSpec.from_tensor(
        values.shape,
        None,
        infinicore.int64,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=values,
    )


def parse_test_cases():
    tests = []

    for B, T, Hk, Hv, Dk, Dv, use_qk_l2norm, strided_qkv in _TEST_CASES:
        q_shape = (B, T, Hk, Dk)
        k_shape = (B, T, Hk, Dk)
        v_shape = (B, T, Hv, Dv)
        gate_shape = (B, T, Hv)
        initial_state_shape = (B, Hv, Dv, Dk)
        pool_size = B * 2 + 3
        state_pool_shape = (pool_size, Hv, Dv, Dk)
        q_strides = strided_bthd_strides(q_shape) if strided_qkv else None
        k_strides = strided_bthd_strides(k_shape) if strided_qkv else None
        v_strides = strided_bthd_strides(v_shape) if strided_qkv else None

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]

            base_inputs = [
                tensor_spec(q_shape, dtype, q_strides),
                tensor_spec(k_shape, dtype, k_strides),
                tensor_spec(v_shape, dtype, v_strides),
                gate_spec(gate_shape),
                beta_spec(gate_shape),
            ]

            tests.append(
                TestCase(
                    inputs=base_inputs
                    + [TensorSpec.from_tensor(initial_state_shape, None, dtype)],
                    kwargs={
                        "mode": "legacy",
                        "use_qk_l2norm": use_qk_l2norm,
                    },
                    description=(
                        f"legacy B={B}, T={T}, Hk={Hk}, Hv={Hv}, "
                        f"Dk={Dk}, Dv={Dv}, dtype={dtype}, "
                        f"strided_qkv={strided_qkv}, l2norm={use_qk_l2norm}"
                    ),
                    tolerance=tol,
                )
            )

            tests.append(
                TestCase(
                    inputs=base_inputs
                    + [
                        TensorSpec.from_tensor(state_pool_shape, None, dtype),
                        index_spec(range(1, B + 1)),
                        index_spec(range(B + 1, 2 * B + 1)),
                    ],
                    kwargs={
                        "mode": "indexed_pool",
                        "use_qk_l2norm": use_qk_l2norm,
                    },
                    output_count=2,
                    description=(
                        f"indexed pool B={B}, T={T}, Hk={Hk}, Hv={Hv}, "
                        f"Dk={Dk}, Dv={Dv}, dtype={dtype}, "
                        f"strided_qkv={strided_qkv}, l2norm={use_qk_l2norm}"
                    ),
                    tolerance=tol,
                )
            )

    for dtype in _TENSOR_DTYPES:
        tests.append(
            TestCase(
                inputs=[
                    tensor_spec((1, 48, 128), dtype),
                    tensor_spec((1, 48, 128), dtype),
                    tensor_spec((1, 48, 128), dtype),
                    gate_spec((1, 1, 48)),
                    beta_spec((1, 1, 48)),
                    TensorSpec.from_tensor((1, 48, 128, 128), None, dtype),
                ],
                kwargs={"mode": "user_3d", "use_qk_l2norm": False},
                description=f"user 3D repro dtype={dtype}",
                tolerance=_TOLERANCE_MAP[dtype],
            )
        )

    return tests


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("recurrent_gated_delta_rule")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, q, k, v, g, beta, initial_state, *args, **kwargs):
        mode = kwargs.pop("mode")
        use_qk_l2norm = kwargs.pop("use_qk_l2norm", False)

        if mode == "legacy":
            out, _ = ref_recurrent_gated_delta_rule(
                q, k, v, g, beta, initial_state, use_qk_l2norm=use_qk_l2norm
            )
            return out

        if mode == "indexed_pool":
            initial_state_indices, final_state_indices = args
            state_pool = initial_state.clone()
            gathered_initial_state = state_pool[initial_state_indices].contiguous()
            out, final_state = ref_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                gathered_initial_state,
                use_qk_l2norm=use_qk_l2norm,
            )
            state_pool[final_state_indices] = final_state.contiguous()
            return out, state_pool

        if mode == "user_3d":
            out, _ = ref_recurrent_gated_delta_rule(
                q.unsqueeze(1),
                k.unsqueeze(1),
                v.unsqueeze(1),
                g,
                beta,
                initial_state,
                use_qk_l2norm=use_qk_l2norm,
            )
            return out

        raise ValueError(f"Unsupported test mode: {mode}")

    def infinicore_operator(self, q, k, v, g, beta, initial_state, *args, **kwargs):
        mode = kwargs.pop("mode")
        use_qk_l2norm = kwargs.pop("use_qk_l2norm", False)

        if mode == "legacy" or mode == "user_3d":
            return infinicore.nn.functional.recurrent_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                use_qk_l2norm=use_qk_l2norm,
            )

        if mode == "indexed_pool":
            initial_state_indices, final_state_indices = args
            out = infinicore.nn.functional.recurrent_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                initial_state_indices=initial_state_indices,
                final_state_indices=final_state_indices,
                use_qk_l2norm=use_qk_l2norm,
            )
            return out, initial_state

        raise ValueError(f"Unsupported test mode: {mode}")


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
