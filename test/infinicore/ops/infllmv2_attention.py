"""
Operator unit tests for InfiniCore InfLLM-V2 attention ops: infllmv2_attention_varlen and infllmv2_attention_kvcache.
Uses the InfiniCore test framework (BaseOperatorTest, TestCase, GenericTestRunner).
Runs only when InfiniCore is built with ENABLE_INFLLMV2 and linked to the infllmv2 .so;
otherwise tests are skipped so CI without the .so still passes.

Run (from InfiniCore dir):
  python test/infinicore/run.py --ops infllmv2_attention --nvidia

Direct:
  python test/infinicore/ops/infllmv2_attention.py --nvidia
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch

from framework import (
    BaseOperatorTest,
    TestCase,
    TensorSpec,
    TensorInitializer,
    GenericTestRunner,
    get_args,
)
from framework.results import CaseResult
from framework.utils.tensor_utils import convert_infinicore_to_torch

# Check for InfLLM-V2 ops; skip entire module if not built
infllmv2_attention_varlen = getattr(infinicore, "infllmv2_attention_varlen", None)
infllmv2_attention_kvcache = getattr(infinicore, "infllmv2_attention_kvcache", None)
INFLLMV2_AVAILABLE = (
    infllmv2_attention_varlen is not None and infllmv2_attention_kvcache is not None
)


def _print_metrics(name, out_infinicore):
    out_t = convert_infinicore_to_torch(out_infinicore)
    l2 = float(out_t.norm())
    max_abs = float(out_t.abs().max())
    print(
        f"  {name}: shape={list(out_infinicore.shape)} L2={l2:.4f} max_abs={max_abs:.4f}"
    )


def _make_varlen_test_case():
    total_q, nheads, head_dim = 8, 2, 8
    total_k, nheads_k = 8, 2
    scale = 1.0 / (head_dim**0.5)
    q_spec = TensorSpec.from_tensor(
        (total_q, nheads, head_dim), None, infinicore.float16
    )
    k_spec = TensorSpec.from_tensor(
        (total_k, nheads_k, head_dim), None, infinicore.float16
    )
    v_spec = TensorSpec.from_tensor(
        (total_k, nheads_k, head_dim), None, infinicore.float16
    )
    cu_q_spec = TensorSpec.from_tensor(
        (3,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    cu_k_spec = TensorSpec.from_tensor(
        (3,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    return TestCase(
        inputs=[q_spec, k_spec, v_spec, cu_q_spec, cu_k_spec],
        kwargs={
            "max_seqlen_q": 4,
            "max_seqlen_k": 4,
            "scale": scale,
            "causal": True,
            "_expected_out_shape": (total_q, nheads, head_dim),
        },
        output_spec=None,
        comparison_target=None,
        tolerance={"atol": 1e-2, "rtol": 1e-2},
        description="InfLLMV2 varlen (2 batches, 4 tokens)",
    )


def _make_varlen_test_case_bf16():
    total_q, nheads, head_dim = 8, 2, 8
    total_k, nheads_k = 8, 2
    scale = 1.0 / (head_dim**0.5)
    q_spec = TensorSpec.from_tensor(
        (total_q, nheads, head_dim), None, infinicore.bfloat16
    )
    k_spec = TensorSpec.from_tensor(
        (total_k, nheads_k, head_dim), None, infinicore.bfloat16
    )
    v_spec = TensorSpec.from_tensor(
        (total_k, nheads_k, head_dim), None, infinicore.bfloat16
    )
    cu_q_spec = TensorSpec.from_tensor(
        (3,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    cu_k_spec = TensorSpec.from_tensor(
        (3,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    return TestCase(
        inputs=[q_spec, k_spec, v_spec, cu_q_spec, cu_k_spec],
        kwargs={
            "max_seqlen_q": 4,
            "max_seqlen_k": 4,
            "scale": scale,
            "causal": True,
            "_expected_out_shape": (total_q, nheads, head_dim),
        },
        output_spec=None,
        comparison_target=None,
        tolerance={"atol": 1e-1, "rtol": 1e-1},
        description="InfLLMV2 varlen BF16 (2 batches, 4 tokens)",
    )


def _make_varlen_test_case_localwindow():
    total_q, nheads, head_dim = 8, 2, 8
    total_k, nheads_k = 8, 2
    scale = 1.0 / (head_dim**0.5)
    q_spec = TensorSpec.from_tensor(
        (total_q, nheads, head_dim), None, infinicore.float16
    )
    k_spec = TensorSpec.from_tensor(
        (total_k, nheads_k, head_dim), None, infinicore.float16
    )
    v_spec = TensorSpec.from_tensor(
        (total_k, nheads_k, head_dim), None, infinicore.float16
    )
    cu_q_spec = TensorSpec.from_tensor(
        (3,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    cu_k_spec = TensorSpec.from_tensor(
        (3,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    return TestCase(
        inputs=[q_spec, k_spec, v_spec, cu_q_spec, cu_k_spec],
        kwargs={
            "max_seqlen_q": 4,
            "max_seqlen_k": 4,
            "scale": scale,
            "causal": False,
            "window_size_left": 2,
            "window_size_right": 0,
            "_expected_out_shape": (total_q, nheads, head_dim),
        },
        output_spec=None,
        comparison_target=None,
        tolerance={"atol": 1e-2, "rtol": 1e-2},
        description="InfLLMV2 varlen local-window (causal=false, left=2, right=0)",
    )


def _make_varlen_test_case_localwindow_left0():
    total_q, nheads, head_dim = 8, 2, 8
    total_k, nheads_k = 8, 2
    scale = 1.0 / (head_dim**0.5)
    q_spec = TensorSpec.from_tensor(
        (total_q, nheads, head_dim), None, infinicore.float16
    )
    k_spec = TensorSpec.from_tensor(
        (total_k, nheads_k, head_dim), None, infinicore.float16
    )
    v_spec = TensorSpec.from_tensor(
        (total_k, nheads_k, head_dim), None, infinicore.float16
    )
    cu_q_spec = TensorSpec.from_tensor(
        (3,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    cu_k_spec = TensorSpec.from_tensor(
        (3,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    return TestCase(
        inputs=[q_spec, k_spec, v_spec, cu_q_spec, cu_k_spec],
        kwargs={
            "max_seqlen_q": 4,
            "max_seqlen_k": 4,
            "scale": scale,
            "causal": False,
            "window_size_left": 0,
            "window_size_right": 0,
            "_expected_out_shape": (total_q, nheads, head_dim),
        },
        output_spec=None,
        comparison_target=None,
        tolerance={"atol": 1e-2, "rtol": 1e-2},
        description="InfLLMV2 varlen local-window (causal=false, left=0, right=0)",
    )


def _make_varlen_test_case_localwindow_left3():
    total_q, nheads, head_dim = 8, 2, 8
    total_k, nheads_k = 8, 2
    scale = 1.0 / (head_dim**0.5)
    q_spec = TensorSpec.from_tensor(
        (total_q, nheads, head_dim), None, infinicore.float16
    )
    k_spec = TensorSpec.from_tensor(
        (total_k, nheads_k, head_dim), None, infinicore.float16
    )
    v_spec = TensorSpec.from_tensor(
        (total_k, nheads_k, head_dim), None, infinicore.float16
    )
    cu_q_spec = TensorSpec.from_tensor(
        (3,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    cu_k_spec = TensorSpec.from_tensor(
        (3,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    return TestCase(
        inputs=[q_spec, k_spec, v_spec, cu_q_spec, cu_k_spec],
        kwargs={
            "max_seqlen_q": 4,
            "max_seqlen_k": 4,
            "scale": scale,
            "causal": False,
            "window_size_left": 3,
            "window_size_right": 0,
            "_expected_out_shape": (total_q, nheads, head_dim),
        },
        output_spec=None,
        comparison_target=None,
        tolerance={"atol": 1e-2, "rtol": 1e-2},
        description="InfLLMV2 varlen local-window (causal=false, left=3, right=0)",
    )


def _make_kvcache_test_case():
    batch, seqlen_q, nheads, head_dim = 1, 1, 2, 8
    cache_len = 4
    scale = 1.0 / (head_dim**0.5)
    q_spec = TensorSpec.from_tensor(
        (batch, seqlen_q, nheads, head_dim), None, infinicore.float16
    )
    k_cache_spec = TensorSpec.from_tensor(
        (batch, cache_len, nheads, head_dim), None, infinicore.float16
    )
    v_cache_spec = TensorSpec.from_tensor(
        (batch, cache_len, nheads, head_dim), None, infinicore.float16
    )
    cache_lens_spec = TensorSpec.from_tensor(
        (batch,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([cache_len], dtype=torch.int32),
    )
    return TestCase(
        inputs=[q_spec, k_cache_spec, v_cache_spec, cache_lens_spec],
        kwargs={
            "scale": scale,
            "causal": True,
            "_expected_out_shape": (batch, seqlen_q, nheads, head_dim),
        },
        output_spec=None,
        comparison_target=None,
        tolerance={"atol": 1e-2, "rtol": 1e-2},
        description="InfLLMV2 kvcache (1 batch, 1 query, cache_len=4)",
    )


def _make_kvcache_test_case_localwindow():
    batch, seqlen_q, nheads, head_dim = 1, 1, 2, 8
    cache_len = 4
    scale = 1.0 / (head_dim**0.5)
    q_spec = TensorSpec.from_tensor(
        (batch, seqlen_q, nheads, head_dim), None, infinicore.float16
    )
    k_cache_spec = TensorSpec.from_tensor(
        (batch, cache_len, nheads, head_dim), None, infinicore.float16
    )
    v_cache_spec = TensorSpec.from_tensor(
        (batch, cache_len, nheads, head_dim), None, infinicore.float16
    )
    cache_lens_spec = TensorSpec.from_tensor(
        (batch,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([cache_len], dtype=torch.int32),
    )
    return TestCase(
        inputs=[q_spec, k_cache_spec, v_cache_spec, cache_lens_spec],
        kwargs={
            "scale": scale,
            "causal": False,
            "window_size_left": 2,
            "window_size_right": 0,
            "_expected_out_shape": (batch, seqlen_q, nheads, head_dim),
        },
        output_spec=None,
        comparison_target=None,
        tolerance={"atol": 1e-2, "rtol": 1e-2},
        description="InfLLMV2 kvcache local-window (causal=false, left=2, right=0)",
    )


def _make_kvcache_test_case_localwindow_left0():
    batch, seqlen_q, nheads, head_dim = 1, 1, 2, 8
    cache_len = 4
    scale = 1.0 / (head_dim**0.5)
    q_spec = TensorSpec.from_tensor(
        (batch, seqlen_q, nheads, head_dim), None, infinicore.float16
    )
    k_cache_spec = TensorSpec.from_tensor(
        (batch, cache_len, nheads, head_dim), None, infinicore.float16
    )
    v_cache_spec = TensorSpec.from_tensor(
        (batch, cache_len, nheads, head_dim), None, infinicore.float16
    )
    cache_lens_spec = TensorSpec.from_tensor(
        (batch,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([cache_len], dtype=torch.int32),
    )
    return TestCase(
        inputs=[q_spec, k_cache_spec, v_cache_spec, cache_lens_spec],
        kwargs={
            "scale": scale,
            "causal": False,
            "window_size_left": 0,
            "window_size_right": 0,
            "_expected_out_shape": (batch, seqlen_q, nheads, head_dim),
        },
        output_spec=None,
        comparison_target=None,
        tolerance={"atol": 1e-2, "rtol": 1e-2},
        description="InfLLMV2 kvcache local-window (causal=false, left=0, right=0)",
    )


def _make_kvcache_test_case_bf16():
    batch, seqlen_q, nheads, head_dim = 1, 1, 2, 8
    cache_len = 4
    scale = 1.0 / (head_dim**0.5)
    q_spec = TensorSpec.from_tensor(
        (batch, seqlen_q, nheads, head_dim), None, infinicore.bfloat16
    )
    k_cache_spec = TensorSpec.from_tensor(
        (batch, cache_len, nheads, head_dim), None, infinicore.bfloat16
    )
    v_cache_spec = TensorSpec.from_tensor(
        (batch, cache_len, nheads, head_dim), None, infinicore.bfloat16
    )
    cache_lens_spec = TensorSpec.from_tensor(
        (batch,),
        None,
        infinicore.int32,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=torch.tensor([cache_len], dtype=torch.int32),
    )
    return TestCase(
        inputs=[q_spec, k_cache_spec, v_cache_spec, cache_lens_spec],
        kwargs={
            "scale": scale,
            "causal": True,
            "_expected_out_shape": (batch, seqlen_q, nheads, head_dim),
        },
        output_spec=None,
        comparison_target=None,
        tolerance={"atol": 1e-1, "rtol": 1e-1},
        description="InfLLMV2 kvcache BF16 (1 batch, 1 query, cache_len=4)",
    )


class InfLLMV2AttentionTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("InfLLMV2Attention")

    def get_test_cases(self):
        if not INFLLMV2_AVAILABLE:
            return []
        return [
            _make_varlen_test_case(),
            _make_kvcache_test_case(),
            _make_kvcache_test_case_localwindow(),
            _make_kvcache_test_case_localwindow_left0(),
            _make_varlen_test_case_localwindow(),
            _make_varlen_test_case_localwindow_left0(),
            _make_varlen_test_case_localwindow_left3(),
            _make_varlen_test_case_bf16(),
            _make_kvcache_test_case_bf16(),
        ]

    def torch_operator(self, *args, **kwargs):
        raise NotImplementedError(
            "InfLLM-V2 has no PyTorch reference in this test (InfiniCore-only)"
        )

    def infinicore_operator(self, *args, **kwargs):
        raise NotImplementedError("InfLLM-V2 uses run_test override (InfiniCore-only)")

    def run_test(self, device, test_case, config):
        test_result = CaseResult(
            success=False,
            return_code=-1,
            test_case=test_case,
            device=device,
        )

        if not INFLLMV2_AVAILABLE:
            test_result.return_code = -2
            test_result.error_message = "infllmv2_attention_varlen/infllmv2_attention_kvcache not available (build without ENABLE_INFLLMV2?)"
            return test_result

        inputs, kwargs = self.prepare_pytorch_inputs_and_kwargs(test_case, device)
        expected_shape = kwargs.pop("_expected_out_shape", None)
        infini_inputs, infini_kwargs, _ = self.prepare_infinicore_inputs_and_kwargs(
            inputs, kwargs, test_case.comparison_target
        )

        if len(infini_inputs) == 5:
            window_size_left = infini_kwargs.get("window_size_left", -1)
            window_size_right = infini_kwargs.get("window_size_right", -1)
            out = infllmv2_attention_varlen(
                infini_inputs[0],
                infini_inputs[1],
                infini_inputs[2],
                infini_inputs[3],
                infini_inputs[4],
                max_seqlen_q=infini_kwargs["max_seqlen_q"],
                max_seqlen_k=infini_kwargs["max_seqlen_k"],
                scale=infini_kwargs["scale"],
                causal=infini_kwargs["causal"],
                window_size_left=window_size_left,
                window_size_right=window_size_right,
            )
            name = "varlen"
        elif len(infini_inputs) == 4:
            window_size_left = infini_kwargs.get("window_size_left", -1)
            window_size_right = infini_kwargs.get("window_size_right", -1)
            out = infllmv2_attention_kvcache(
                infini_inputs[0],
                infini_inputs[1],
                infini_inputs[2],
                infini_inputs[3],
                scale=infini_kwargs["scale"],
                causal=infini_kwargs["causal"],
                window_size_left=window_size_left,
                window_size_right=window_size_right,
            )
            name = "kvcache"
        else:
            test_result.error_message = (
                f"Unexpected number of inputs: {len(infini_inputs)}"
            )
            return test_result

        infinicore.sync_stream()

        if out is None:
            test_result.error_message = "InfiniCore operator returned None"
            return test_result

        shape = out.shape
        if expected_shape is not None and tuple(shape) != tuple(expected_shape):
            test_result.error_message = (
                f"Shape mismatch: got {list(shape)}, expected {list(expected_shape)}"
            )
            return test_result

        out_t = convert_infinicore_to_torch(out)
        if torch.isnan(out_t).any() or torch.isinf(out_t).any():
            test_result.error_message = "Output contained NaN/Inf"
            return test_result

        _print_metrics(name, out)
        test_result.success = True
        test_result.return_code = 0
        return test_result


def main():
    args = get_args()
    if not args.nvidia:
        print("InfLLM-V2 ops require CUDA; use --nvidia to run on GPU.")
        sys.exit(0)
    if not INFLLMV2_AVAILABLE:
        print(
            "infllmv2_attention_varlen / infllmv2_attention_kvcache not available. Build InfiniCore with --aten=y --infllmv2=..."
        )
        sys.exit(0)

    runner = GenericTestRunner(InfLLMV2AttentionTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
