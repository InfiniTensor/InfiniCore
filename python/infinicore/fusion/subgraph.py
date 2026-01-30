"""
子图表示模块 - 轻量级、可哈希的子图数据结构

设计原则：
1. 解耦：不依赖 ninetoothed 或 torch.fx
2. 缓存友好：frozen dataclass 支持 __hash__ 和 __eq__
3. 序列化：易于打印日志和调试
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict
import hashlib


@dataclass(frozen=True)
class OpNode:
    """
    算子节点（不可变，用于缓存 Key）
    
    Attributes:
        op_type: 算子类型标识符（如 "rms_norm", "silu", "mul"）
        inputs: 输入张量名元组
        outputs: 输出张量名元组
        attrs: 算子属性（元组化以支持哈希）
    
    Example:
        >>> node = OpNode(
        ...     op_type="silu",
        ...     inputs=("x",),
        ...     outputs=("y",),
        ... )
    """
    op_type: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    attrs: Optional[Tuple[Tuple[str, Any], ...]] = None

    def __hash__(self) -> int:
        return hash((self.op_type, self.inputs, self.outputs, self.attrs))

    def __repr__(self) -> str:
        return f"OpNode({self.op_type}, inputs={self.inputs}, outputs={self.outputs})"


@dataclass(frozen=True)
class SubGraph:
    """
    子图表示（不可变，用于缓存 Key）
    
    表示一个可融合的算子序列，节点按拓扑排序排列。
    
    Attributes:
        nodes: 拓扑排序的算子节点元组
        input_names: 子图外部输入名
        output_names: 子图外部输出名
    
    Example:
        >>> graph = SubGraph(
        ...     nodes=(
        ...         OpNode("silu", ("x",), ("y1",)),
        ...         OpNode("mul", ("y1", "x"), ("y2",)),
        ...     ),
        ...     input_names=("x",),
        ...     output_names=("y2",),
        ... )
    """
    nodes: Tuple[OpNode, ...]
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...]

    def __hash__(self) -> int:
        return hash((self.nodes, self.input_names, self.output_names))

    def cache_key(
        self,
        input_dtypes: Dict[str, str],
        input_shapes: Dict[str, Tuple[int, ...]]
    ) -> str:
        """
        生成缓存 Key（包含图结构 + dtype + shape）。
        
        不同的 dtype 或 shape 组合会生成不同的内核，因此需要包含在缓存键中。
        
        Args:
            input_dtypes: 输入张量的数据类型字典
            input_shapes: 输入张量的形状字典
            
        Returns:
            16 字符的十六进制哈希字符串
        """
        key_data = (
            hash(self),
            tuple(sorted(input_dtypes.items())),
            tuple((k, v) for k, v in sorted(input_shapes.items()))
        )
        return hashlib.sha256(str(key_data).encode()).hexdigest()[:16]

    def __len__(self) -> int:
        """返回子图中的节点数"""
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"SubGraph(nodes={len(self.nodes)}, inputs={self.input_names}, outputs={self.output_names})"
