"""
内核编译器模块 - 封装 ninetoothed fusion 编译能力

将 InfiniCore 的 SubGraph 表示转换为 ninetoothed 可处理的格式，
调用 fusion 模块进行算子融合编译。
"""

from typing import Dict, Tuple, Callable, Optional, Any
import functools

from infinicore.fusion.subgraph import SubGraph, OpNode
from infinicore.fusion.fusion_config import FusionConfig


class FusionError(Exception):
    """融合编译失败异常"""
    pass


class CompiledKernel:
    """
    编译后的融合内核封装
    
    Attributes:
        kernel: 可调用的融合内核函数
        graph: 原始子图
        cache_key: 缓存键
    """
    
    def __init__(
        self,
        kernel: Callable,
        graph: SubGraph,
        cache_key: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.kernel = kernel
        self.graph = graph
        self.cache_key = cache_key
        self.metadata = metadata or {}
    
    def __call__(self, *args, **kwargs):
        return self.kernel(*args, **kwargs)
    
    def __repr__(self) -> str:
        return f"CompiledKernel(key={self.cache_key}, nodes={len(self.graph)})"


# 算子注册表：映射算子名到 (premake_function, requires_special_handling)
# 对于标准逐元素算子，只需要 ndim；对于特殊算子需要额外参数
_OP_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _init_op_registry():
    """延迟初始化算子注册表，避免循环导入"""
    global _OP_REGISTRY
    if _OP_REGISTRY:
        return
    
    try:
        import ntops.kernels.silu
        import ntops.kernels.gelu
        import ntops.kernels.relu
        import ntops.kernels.sigmoid
        import ntops.kernels.mul
        import ntops.kernels.add
        import ntops.kernels.sub
        import ntops.kernels.div
        import ntops.kernels.rms_norm
        
        _OP_REGISTRY = {
            # 逐元素激活函数 (input, output)
            "silu": {
                "premake": ntops.kernels.silu.premake,
                "type": "unary",  # 单输入单输出
            },
            "gelu": {
                "premake": ntops.kernels.gelu.premake,
                "type": "unary",
            },
            "relu": {
                "premake": ntops.kernels.relu.premake,
                "type": "unary",
            },
            "sigmoid": {
                "premake": ntops.kernels.sigmoid.premake,
                "type": "unary",
            },
            # 逐元素二元算子 (input, other, output)
            "mul": {
                "premake": ntops.kernels.mul.premake,
                "type": "binary",
            },
            "add": {
                "premake": ntops.kernels.add.premake,
                "type": "binary",
            },
            "sub": {
                "premake": ntops.kernels.sub.premake,
                "type": "binary",
            },
            "div": {
                "premake": ntops.kernels.div.premake,
                "type": "binary",
            },
            # RMSNorm (input, weight, eps, output, num_normalized_elements)
            "rms_norm": {
                "premake": ntops.kernels.rms_norm.premake,
                "type": "rms_norm",  # 特殊处理
            },
        }
    except ImportError as e:
        # 如果 ntops 不可用，注册表保持为空
        pass


def get_supported_fusion_ops() -> set:
    """返回支持融合的算子集合"""
    _init_op_registry()
    return set(_OP_REGISTRY.keys())


class KernelCompiler:
    """
    内核编译器 - 封装 ninetoothed fusion 编译能力
    
    职责：
    1. 将 SubGraph 转换为 ninetoothed 可处理的格式
    2. 调用 ninetoothed.fusion 进行算子融合
    3. 返回编译后的可调用内核
    """
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self._ntops_available = False
        self._ninetoothed_available = False
        self._init_backends()
    
    def _init_backends(self):
        """初始化后端依赖"""
        try:
            import ntops
            import ninetoothed
            self._ntops = ntops
            self._ninetoothed = ninetoothed
            self._ntops_available = True
            self._ninetoothed_available = True
            _init_op_registry()
        except ImportError as e:
            if self.config.debug_mode:
                print(f"[KernelCompiler] Backend not available: {e}")
    
    @property
    def is_available(self) -> bool:
        """检查编译器后端是否可用"""
        return self._ntops_available and self._ninetoothed_available and bool(_OP_REGISTRY)
    
    def compile(
        self,
        graph: SubGraph,
        input_dtypes: Dict[str, str],
        input_shapes: Dict[str, Tuple[int, ...]]
    ) -> CompiledKernel:
        """
        将子图编译为融合内核。
        
        Args:
            graph: 子图描述
            input_dtypes: 输入张量的数据类型
            input_shapes: 输入张量的形状
            
        Returns:
            CompiledKernel: 编译后的融合内核
            
        Raises:
            FusionError: 编译失败时抛出
        """
        if not self.is_available:
            raise FusionError("Backend not available: ntops or ninetoothed not installed")
        
        cache_key = graph.cache_key(input_dtypes, input_shapes)
        
        if self.config.debug_mode:
            print(f"[KernelCompiler] Compiling graph: {graph}")
            print(f"[KernelCompiler] Cache key: {cache_key}")
        
        try:
            # 推断张量维度
            ndim = self._infer_tensor_ndim(input_shapes)
            
            # Step 1: 为每个算子创建 _Handle 对象
            handles = self._create_handles_for_graph(graph, ndim)
            
            # Step 2: 构建 ninetoothed Node 列表
            nodes = self._build_fusion_nodes(handles, graph)
            
            # Step 3: 调用融合
            from ninetoothed.fusion import _fuse_nodes
            fused_nodes = _fuse_nodes(nodes)
            
            if len(fused_nodes) != 1:
                raise FusionError(f"Fusion produced {len(fused_nodes)} nodes, expected 1")
            
            fused_kernel = fused_nodes[0].kernel
            
            return CompiledKernel(
                kernel=fused_kernel,
                graph=graph,
                cache_key=cache_key,
                metadata={
                    "input_dtypes": input_dtypes,
                    "input_shapes": input_shapes,
                    "num_original_nodes": len(graph.nodes),
                }
            )
            
        except Exception as e:
            raise FusionError(f"Compilation failed: {e}") from e
    
    def _infer_tensor_ndim(self, input_shapes: Dict[str, Tuple[int, ...]]) -> int:
        """从输入形状推断张量维度"""
        if not input_shapes:
            return 2  # 默认 2D
        
        # 取第一个非空形状的维度
        for name, shape in input_shapes.items():
            if shape:
                return len(shape)
        
        return 2
    
    def _create_handle_for_op(self, op_type: str, ndim: int) -> Any:
        """
        为单个算子创建 ninetoothed 内核句柄 (_Handle)
        
        Args:
            op_type: 算子类型名称
            ndim: 张量维度
            
        Returns:
            _Handle 对象，可用于融合
        """
        if op_type not in _OP_REGISTRY:
            raise FusionError(f"Operator '{op_type}' not in fusion registry")
        
        op_info = _OP_REGISTRY[op_type]
        premake_fn = op_info["premake"]
        op_kind = op_info["type"]
        
        # 根据算子类型调用不同的 premake 签名
        if op_kind in ("unary", "binary"):
            # 标准逐元素算子: premake(ndim)
            arrangement, application, tensors = premake_fn(ndim)
        elif op_kind == "rms_norm":
            # RMSNorm: premake(ndim, num_normalized_dims)
            # 对于 LLM，通常归一化最后一个维度
            arrangement, application, tensors = premake_fn(ndim, num_normalized_dims=1)
        else:
            raise FusionError(f"Unknown operator kind: {op_kind}")
        
        # 调用 ninetoothed.make 创建 _Handle
        handle = self._ninetoothed.make(
            arrangement,
            application,
            tensors,
            num_warps=4,
            num_stages=2,
        )
        
        if self.config.debug_mode:
            print(f"[KernelCompiler] Created handle for {op_type}: {handle}")
        
        return handle
    
    def _create_handles_for_graph(self, graph: SubGraph, ndim: int) -> Dict[str, Any]:
        """为图中所有算子创建 _Handle 对象"""
        handles = {}
        for op_node in graph.nodes:
            if op_node.op_type not in handles:
                handles[op_node.op_type] = self._create_handle_for_op(op_node.op_type, ndim)
        return handles
    
    def _build_fusion_nodes(self, handles: Dict[str, Any], graph: SubGraph) -> list:
        """
        构建 ninetoothed fusion 可处理的 Node 列表
        
        将每个 OpNode 包装为 ninetoothed.fusion.Node，
        Node 持有 _Handle 对象和运行时参数信息。
        """
        from ninetoothed.fusion import Node
        
        nodes = []
        for op_node in graph.nodes:
            handle = handles[op_node.op_type]
            
            # Node 需要 args 来建立数据依赖关系
            # 传入 inputs + outputs 作为参数，让 ninetoothed 能识别共享的数据对象（这里是字符串名）
            node_args = list(op_node.inputs) + list(op_node.outputs)
            node = Node(handle, args=tuple(node_args), kwargs={})
            nodes.append(node)
        
        return nodes
