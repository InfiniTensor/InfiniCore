"""
LLM 推理常用融合模式

定义大语言模型推理中常见的可融合算子组合。
"""

from typing import List

from infinicore.fusion.subgraph import SubGraph, OpNode


def create_swiglu_pattern() -> SubGraph:
    """
    创建 SwiGLU 激活融合模式
    
    SwiGLU = SiLU(gate) * up
    
    常见于 LLaMA、Mistral 等模型的 FFN 层。
    """
    return SubGraph(
        nodes=(
            OpNode(
                op_type="silu",
                inputs=("gate",),
                outputs=("gate_activated",),
            ),
            OpNode(
                op_type="mul",
                inputs=("gate_activated", "up"),
                outputs=("output",),
            ),
        ),
        input_names=("gate", "up"),
        output_names=("output",),
    )


def create_add_rms_norm_pattern() -> SubGraph:
    """
    创建 残差连接 + RMSNorm 融合模式
    
    output = rms_norm(x + residual, weight)
    
    常见于 Transformer 层的后处理。
    """
    return SubGraph(
        nodes=(
            OpNode(
                op_type="add",
                inputs=("x", "residual"),
                outputs=("sum",),
            ),
            OpNode(
                op_type="rms_norm",
                inputs=("sum", "weight"),
                outputs=("output",),
            ),
        ),
        input_names=("x", "residual", "weight"),
        output_names=("output",),
    )


def create_gelu_pattern() -> SubGraph:
    """
    创建 GELU 激活模式（单算子，用于测试）
    """
    return SubGraph(
        nodes=(
            OpNode(
                op_type="gelu",
                inputs=("x",),
                outputs=("output",),
            ),
        ),
        input_names=("x",),
        output_names=("output",),
    )


# 预定义的 LLM 融合模式列表
LLM_FUSION_PATTERNS: List[SubGraph] = [
    create_swiglu_pattern(),
    create_add_rms_norm_pattern(),
]
