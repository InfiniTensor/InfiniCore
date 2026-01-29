"""
Graph 转换器模块 - 将 InfiniCore 录制的 Graph 转换为 FusionScheduler 可处理的 SubGraph

这个模块是 InfiniLM 集成的关键桥梁：
1. InfiniLM 推理时使用 start/stop_graph_recording() 捕获算子调用
2. 本模块将录制的 Graph 转换为 SubGraph
3. FusionScheduler 分析 SubGraph 决定是否融合执行
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from infinicore.fusion.subgraph import SubGraph, OpNode


@dataclass
class GraphOpInfo:
    """录制图中算子的信息"""
    op_type: str
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...]
    attrs: Optional[Dict[str, Any]] = None


def convert_graph_to_subgraph(graph) -> Optional[SubGraph]:
    """
    将 InfiniCore 录制的 Graph 转换为 SubGraph。
    
    使用 C++ Graph.operators() 接口直接提取算子信息。
    
    Args:
        graph: infinicore.Graph 对象，从 stop_graph_recording() 获取
        
    Returns:
        SubGraph: 可供 FusionScheduler 处理的子图描述
        None: 如果图为空或无法转换
    """
    if graph is None:
        return None
    
    # 使用新接口：Graph.operators() 返回 GraphOperator 列表
    if not hasattr(graph, 'operators'):
        # 旧版本 InfiniCore，使用 fallback 逻辑
        return _convert_graph_legacy(graph)
    
    try:
        operators = graph.operators()
    except Exception:
        return None
    
    if not operators or len(operators) == 0:
        return None
    
    nodes = []
    all_tensor_names = set()
    
    for i, op in enumerate(operators):
        # 获取算子类型（如 "Gemm" -> "gemm"）
        op_type = op.op_type.lower() if op.op_type else f"op_{i}"
        
        # 从 tensor_metas 提取输入输出
        inputs = []
        outputs = []
        
        for j, meta in enumerate(op.tensor_metas):
            name = f"t_{i}_{j}"
            all_tensor_names.add(name)
            if meta.is_input:
                inputs.append(name)
            else:
                outputs.append(name)
        
        # 如果没有捕获到任何张量，使用占位符
        if not inputs and not outputs:
            inputs = [f"input_{i}"]
            outputs = [f"output_{i}"]
        
        nodes.append(OpNode(
            op_type=op_type,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
        ))
    
    # 推断子图的外部输入/输出
    if nodes:
        graph_inputs = nodes[0].inputs
        graph_outputs = nodes[-1].outputs
    else:
        graph_inputs = ()
        graph_outputs = ()
    
    return SubGraph(
        nodes=tuple(nodes),
        input_names=graph_inputs,
        output_names=graph_outputs,
    )


def _convert_graph_legacy(graph) -> Optional[SubGraph]:
    """Fallback: 旧版本 Graph 对象的转换逻辑"""
    underlying = getattr(graph, '_graph', None)
    if underlying is None:
        return None
    
    ops_info = _extract_ops_from_graph(underlying)
    if not ops_info:
        return None
    
    return _build_subgraph_from_ops(ops_info)


def _extract_ops_from_graph(underlying_graph) -> List[GraphOpInfo]:
    """
    从底层 Graph 对象提取算子信息。
    
    Args:
        underlying_graph: C++ Graph 对象 (_infinicore.Graph)
        
    Returns:
        算子信息列表
        
    Note:
        当前这是一个占位实现。完整实现需要：
        1. C++ Graph 类添加 nodes() 或 get_operations() 方法
        2. 通过 pybind11 暴露到 Python
    """
    ops_info = []
    
    # 检查是否有 get_nodes 或类似方法
    if hasattr(underlying_graph, 'get_nodes'):
        for node in underlying_graph.get_nodes():
            op_info = GraphOpInfo(
                op_type=node.op_type,
                input_names=tuple(node.inputs),
                output_names=tuple(node.outputs),
                attrs=dict(node.attrs) if hasattr(node, 'attrs') else None
            )
            ops_info.append(op_info)
    elif hasattr(underlying_graph, 'nodes'):
        # 另一种可能的接口
        for node in underlying_graph.nodes:
            op_info = GraphOpInfo(
                op_type=str(node.op_type),
                input_names=tuple(str(i) for i in node.inputs),
                output_names=tuple(str(o) for o in node.outputs),
            )
            ops_info.append(op_info)
    else:
        # Graph 没有暴露节点信息
        # TODO: 需要扩展 C++ pybind11 接口
        pass
    
    return ops_info


def _build_subgraph_from_ops(ops_info: List[GraphOpInfo]) -> Optional[SubGraph]:
    """
    从算子信息列表构建 SubGraph。
    
    Args:
        ops_info: 算子信息列表
        
    Returns:
        SubGraph 对象
    """
    if not ops_info:
        return None
    
    # 构建 OpNode 列表
    nodes = []
    for op in ops_info:
        node = OpNode(
            op_type=op.op_type,
            inputs=op.input_names,
            outputs=op.output_names,
            attrs=tuple(op.attrs.items()) if op.attrs else None
        )
        nodes.append(node)
    
    # 推断输入和输出名称
    all_inputs = set()
    all_outputs = set()
    
    for op in ops_info:
        all_inputs.update(op.input_names)
        all_outputs.update(op.output_names)
    
    # 图的真实输入 = 不是任何算子输出的输入
    graph_inputs = tuple(sorted(all_inputs - all_outputs))
    # 图的真实输出 = 最后一个算子的输出
    graph_outputs = ops_info[-1].output_names if ops_info else ()
    
    return SubGraph(
        nodes=tuple(nodes),
        input_names=graph_inputs,
        output_names=graph_outputs,
    )


# ============================================================
# 模式匹配辅助函数
# ============================================================

def match_fusion_pattern(graph: SubGraph, pattern: SubGraph) -> bool:
    """
    检查子图是否匹配指定的融合模式。
    
    Args:
        graph: 待匹配的子图
        pattern: 融合模式模板 (如 SwiGLU 模式)
        
    Returns:
        True 如果匹配，否则 False
    """
    if len(graph.nodes) != len(pattern.nodes):
        return False
    
    for g_node, p_node in zip(graph.nodes, pattern.nodes):
        if g_node.op_type != p_node.op_type:
            return False
    
    return True


def find_fusable_subgraphs(
    graph: SubGraph,
    patterns: List[SubGraph]
) -> List[Tuple[int, int, SubGraph]]:
    """
    在图中查找所有可融合的子图。
    
    Args:
        graph: 完整的计算图
        patterns: 融合模式列表
        
    Returns:
        列表，每个元素是 (起始索引, 结束索引, 匹配的模式)
    """
    results = []
    
    for pattern in patterns:
        pattern_len = len(pattern.nodes)
        
        for start_idx in range(len(graph.nodes) - pattern_len + 1):
            # 提取子图片段
            sub_nodes = graph.nodes[start_idx:start_idx + pattern_len]
            sub_graph = SubGraph(
                nodes=sub_nodes,
                input_names=graph.input_names,  # 简化处理
                output_names=graph.output_names,
            )
            
            if match_fusion_pattern(sub_graph, pattern):
                results.append((start_idx, start_idx + pattern_len, pattern))
    
    return results
