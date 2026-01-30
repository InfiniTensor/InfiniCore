"""
融合调度配置模块
"""

from dataclasses import dataclass


@dataclass
class FusionConfig:
    """
    融合调度配置
    
    控制融合行为的运行时参数，支持动态开关和调优。
    
    Attributes:
        enable_fusion: 总开关，False 时所有子图回退到标准执行
        enable_cache: 内核缓存开关，关闭后每次都重新编译
        max_graph_size: 最大子图节点数，超过此值不尝试融合
        fallback_on_error: 融合失败时是否回退到标准执行
        debug_mode: 调试模式，开启后打印融合决策信息
        min_tensor_elements: V1启发式规则 - 最小张量元素数阈值
        min_nodes_for_fusion: V1启发式规则 - 最少节点数才尝试融合
    
    Example:
        >>> config = FusionConfig(enable_fusion=True, debug_mode=True)
        >>> scheduler = FusionScheduler(config)
    """
    # 核心开关
    enable_fusion: bool = True
    enable_cache: bool = True
    fallback_on_error: bool = True
    debug_mode: bool = False
    
    # 图大小限制
    max_graph_size: int = 10
    
    # V1 静态启发式规则参数
    min_tensor_elements: int = 1024
    min_nodes_for_fusion: int = 2

    def __repr__(self) -> str:
        return (
            f"FusionConfig("
            f"enable_fusion={self.enable_fusion}, "
            f"min_elements={self.min_tensor_elements}, "
            f"min_nodes={self.min_nodes_for_fusion})"
        )
