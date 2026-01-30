"""
Patterns 模块初始化
"""

from infinicore.fusion.patterns.llm_patterns import (
    create_swiglu_pattern,
    create_add_rms_norm_pattern,
    LLM_FUSION_PATTERNS,
)

__all__ = [
    "create_swiglu_pattern",
    "create_add_rms_norm_pattern",
    "LLM_FUSION_PATTERNS",
]
