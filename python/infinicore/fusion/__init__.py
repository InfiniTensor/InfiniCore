"""
InfiniCore Fusion Module - 运行时算子融合调度器

提供基于 ninetoothed/ntops 的自动算子融合能力，支持：
- 子图描述和缓存
- 静态启发式融合决策
- 运行时开关配置
"""

from infinicore.fusion.subgraph import OpNode, SubGraph
from infinicore.fusion.fusion_config import FusionConfig
from infinicore.fusion.heuristics import FusionHeuristics
from infinicore.fusion.fusion_scheduler import FusionScheduler
from infinicore.fusion.graph_converter import (
    convert_graph_to_subgraph,
    match_fusion_pattern,
    find_fusable_subgraphs,
    GraphOpInfo,
)

__all__ = [
    "OpNode",
    "SubGraph",
    "FusionConfig",
    "FusionHeuristics",
    "FusionScheduler",
    "convert_graph_to_subgraph",
    "match_fusion_pattern",
    "find_fusable_subgraphs",
    "GraphOpInfo",
]
