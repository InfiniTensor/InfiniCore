# Internal one-shot helper (not imported at runtime): regenerate fused_moe.py from vLLM.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[5]
    src = (
        repo
        / ".venv-vllm/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/fused_moe.py"
    )
    dst = Path(__file__).resolve().parent / "fused_moe.py"
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    out = "".join(lines[:1896])

    reps: list[tuple[str, str]] = [
        ("import vllm.envs as envs", "from . import envs"),
        (
            "import vllm.model_executor.layers.fused_moe.modular_kernel as mk\n",
            "",
        ),
        ("from vllm import _custom_ops as ops\n", "from . import ops_shim as ops\n"),
        ("from vllm.logger import init_logger", "from .logging_utils import init_logger"),
        (
            "from vllm.model_executor.layers.fused_moe.activation import",
            "from .activation import",
        ),
        (
            "from vllm.model_executor.layers.fused_moe.config import (\n"
            "    FUSED_MOE_UNQUANTIZED_CONFIG,\n"
            "    FusedMoEConfig,\n"
            "    FusedMoEParallelConfig,\n"
            "    FusedMoEQuantConfig,\n"
            "    _get_config_dtype_str,\n)",
            "from .config_light import (\n"
            "    FUSED_MOE_UNQUANTIZED_CONFIG,\n"
            "    FusedMoEQuantConfig,\n"
            "    _get_config_dtype_str,\n)",
        ),
        (
            "from vllm.model_executor.layers.fused_moe.moe_align_block_size import",
            "from .moe_align_block_size import",
        ),
        (
            "from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (\n"
            "    TopKWeightAndReduceNoOP,\n)\n",
            "",
        ),
        (
            "from vllm.model_executor.layers.fused_moe.utils import",
            "from .utils_moe import",
        ),
        (
            "from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4\n",
            "",
        ),
        (
            "from vllm.model_executor.layers.quantization.utils.mxfp6_utils import dequant_mxfp6\n",
            "",
        ),
        (
            "from vllm.model_executor.layers.quantization.utils.quant_utils import (\n"
            "    QuantKey,\n"
            "    kFp8Dynamic128Sym,\n"
            "    kFp8DynamicTensorSym,\n"
            "    kFp8DynamicTokenSym,\n"
            "    kFp8Static128BlockSym,\n"
            "    kFp8StaticChannelSym,\n"
            "    kFp8StaticTensorSym,\n)\n",
            "",
        ),
        ("from vllm.platforms import current_platform", "from .platform_ctx import current_platform"),
        ("from vllm.triton_utils import tl, triton", "import triton\nimport triton.language as tl"),
        (
            "from vllm.utils.torch_utils import direct_register_custom_op",
            "from .torch_register import direct_register_custom_op, infinilm_fused_lib",
        ),
    ]
    for a, b in reps:
        if a not in out:
            raise SystemExit(f"missing pattern fragment:\n{a[:200]}")
        out = out.replace(a, b)

    out = out.replace(
        "activation_out_dim = mk.FusedMoEExpertsModular.adjust_N_for_activation(\n"
        "        N, activation_enum\n"
        "    )",
        "activation_out_dim = N if not activation_enum.is_gated else N // 2",
    )

    out = out.replace(
        "    from vllm.model_executor.layers.fused_moe import get_config\n\n"
        "    override_config = get_config()",
        "    override_config = None",
    )

    out = out.replace("torch.ops.vllm.", "torch.ops.infinilm.")

    out = out.replace(
        "direct_register_custom_op(\n    op_name=\"inplace_fused_experts\",",
        "direct_register_custom_op(\n    op_name=\"inplace_fused_experts\",\n"
        "    target_lib=infinilm_fused_lib,",
    )
    out = out.replace(
        "direct_register_custom_op(\n    op_name=\"outplace_fused_experts\",",
        "direct_register_custom_op(\n    op_name=\"outplace_fused_experts\",\n"
        "    target_lib=infinilm_fused_lib,",
    )

    needle = "def dispatch_fused_moe_kernel(\n"
    idx = out.find(needle)
    if idx == -1:
        raise SystemExit("dispatch_fused_moe_kernel not found")
    insert_at = idx + len(needle)
    # After opening def line + newline, insert guard at body start
    guard = (
        "    if (use_int8_w8a16 or use_int4_w4a16) and (\n"
        "        block_shape is not None and block_shape[1] > 0\n"
        "    ):\n"
        "        raise NotImplementedError(\n"
        "            \"InfiniLM vendor fused_moe: INT4/INT8 WNA16 CUDA path requires vLLM native ops.\"\n"
        "        )\n\n"
    )
    # Find first line after def that's already indented (skip docstring? none)
    # Insert right after `) -> None:` line end - actually after `):`
    line_end = out.find(") -> None:", idx)
    if line_end == -1:
        line_end = out.find("):", idx)
    body_start = out.find("\n", line_end) + 1
    out = out[:body_start] + guard + out[body_start:]

    ocp_start = out.find("    if ocp_mx_scheme is not None:\n        # TODO: On platforms")
    if ocp_start == -1:
        raise SystemExit("ocp_mx block not found")
    ocp_end = out.find("\n    qhidden_states, a1q_scale = moe_kernel_quantize_input(", ocp_start)
    if ocp_end == -1:
        raise SystemExit("ocp_mx block end not found")
    out = (
        out[:ocp_start]
        + "    if ocp_mx_scheme is not None:\n"
        "        raise NotImplementedError(\n"
        "            \"InfiniLM vendor fused_moe: ocp_mx_scheme is unsupported.\"\n"
        "        )\n\n"
        + out[ocp_end + 1 :]
    )

    hdr = (
        '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"""Fused MoE Triton kernels (InfiniLM vendor; see NOTICE)."""\n\n'
    )
    # Docstring already in file - replace opening docstring line
    out = out.replace(
        '"""Fused MoE Triton kernels."""\n\n',
        '"""Fused MoE Triton kernels (InfiniLM vendor; see NOTICE)."""\n\n',
        1,
    )

    dst.write_text(out, encoding="utf-8")
    print("wrote", dst)


if __name__ == "__main__":
    main()
