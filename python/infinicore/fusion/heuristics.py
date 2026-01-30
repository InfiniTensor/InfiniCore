import json
import os
from typing import Dict, Tuple, Set, Optional, Any

from infinicore.fusion.subgraph import SubGraph
from infinicore.fusion.fusion_config import FusionConfig


def _detect_hardware_environment() -> str:
    """
    检测当前硬件环境
    
    Returns:
        "muxi" | "tianshu" | "default"
    """
    try:
        from infinicore.lib import _infinicore
        
        all_device_types = tuple(_infinicore.Device.Type.__members__.values())[:-1]
        all_device_count = tuple(_infinicore.get_device_count(dt) for dt in all_device_types)
        
        for device_type, count in zip(all_device_types, all_device_count):
            if count > 0:
                if device_type == _infinicore.Device.Type.MOORE:
                    return "muxi"
                elif device_type == _infinicore.Device.Type.METAX:
                    return "tianshu"
        return "default"
    except Exception:
        return "default"


def _get_supported_ops() -> Set[str]:
    """获取支持融合的算子集合，与 kernel_compiler 保持同步"""
    fallback_ops = {
        "silu", "gelu", "relu", "sigmoid",
        "add", "mul", "sub", "div",
        "rms_norm", "layer_norm",
    }

    try:
        from infinicore.fusion.kernel_compiler import get_supported_fusion_ops
        ops = get_supported_fusion_ops()
        # 如果 kernel_compiler 返回空集（ntops 不可用），使用 fallback
        return ops if ops else fallback_ops
    except ImportError:
        return fallback_ops


# V1 支持融合的算子类型（延迟初始化）
SUPPORTED_OPS: Set[str] = set()


class FusionHeuristics:
    """
    静态启发式规则 - 决定是否值得融合

    V1 实现基于简单规则过滤：
    1. 节点数检查
    2. 张量大小检查
    3. 算子类型检查
    4. profile 决策（unfused 总时间 vs fused 时间）

    profile 缺失/异常：打印错误并返回 False
    """

    def __init__(self, config: FusionConfig, profile_path: Optional[str]):
        self.config = config
        self._supported_ops: Optional[Set[str]] = None
        self._profile_cache: Optional[Dict[str, Any]] = None
        self._profile_path_cached: Optional[str] = None
        
        # 自动检测硬件环境并构建 profile 路径
        env = _detect_hardware_environment()
        profile_dir = os.path.join(os.path.dirname(__file__), "profile_result")
        
        if env == "muxi":
            self.profile_path = os.path.join(profile_dir, "profile_result_muxi.json")
        elif env == "tianshu":
            self.profile_path = os.path.join(profile_dir, "profile_result_tianshu.json")
        else:
            self.profile_path = os.path.join(profile_dir, "default.json")

    def _get_ops(self) -> Set[str]:
        """获取支持的算子集合（带缓存）"""
        if self._supported_ops is None:
            self._supported_ops = _get_supported_ops()
        return self._supported_ops

    # ---------------- profile helpers ----------------

    def _load_profile(self) -> Dict[str, Any]:
        """
        加载 profile 数据（带缓存）

        Returns:
            {"unfused": {...}, "fused": {...}}
        """
        # cache hit
        if self._profile_cache is not None and self._profile_path_cached == self.profile_path:
            return self._profile_cache

        if not os.path.exists(self.profile_path):
            raise FileNotFoundError(f"Profile not found: {self.profile_path}")

        with open(self.profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "unfused" not in data or "fused" not in data:
            raise ValueError("Profile must contain 'unfused' and 'fused' keys")

        self._profile_cache = data
        self._profile_path_cached = self.profile_path
        return data

    def _shape_to_key(self, shape: Tuple[int, ...]) -> str:
        """tuple -> '[1, 512, 4096]'"""
        return "[" + ", ".join(str(int(x)) for x in shape) + "]"

    def _pick_profile_shape_key(
        self,
        graph: SubGraph,
        input_shapes: Dict[str, Tuple[int, ...]]
    ) -> Optional[str]:
        """
        选择代表性的 shape key 用于查询 profile
        优先使用 graph.input_names 以避免选到权重
        """
        for name in graph.input_names:
            shape = input_shapes.get(name)
            if shape:
                return self._shape_to_key(shape)

        for _, shape in input_shapes.items():
            if shape:
                return self._shape_to_key(shape)

        return None

    def _fused_op_key(self, graph: SubGraph) -> str:
        """生成融合算子 key，如 'add+rms_norm'"""
        return "+".join(node.op_type for node in graph.nodes)

    def _parse_shape_key(self, shape_key: str) -> Tuple[int, ...]:
        """'[1, 512, 4096]' -> (1, 512, 4096)"""
        return tuple(int(x.strip()) for x in shape_key.strip("[]").split(","))
    
    def _lookup_nearest_shape(
        self,
        shape_key: str,
        shape_map: Dict[str, Any],
    ) -> Optional[Any]:
        """
        查找最接近的 shape bucket
        规则：rank、B、H 必须一致，token(S) 维度做 lower-bound
        """
        target = self._parse_shape_key(shape_key)
        best_key = None
        best_s = None
    
        for k in shape_map.keys():
            try:
                cand = self._parse_shape_key(k)
            except Exception:
                continue
    
            if len(cand) != len(target):
                continue
    
            # B 和 H 必须完全一致
            if cand[0] != target[0] or cand[-1] != target[-1]:
                continue
    
            s = cand[1]
            if s <= target[1]:
                if best_s is None or s > best_s:
                    best_s = s
                    best_key = k
    
        # 如果所有 bucket 的 S 都 > target.S，取最小的
        if best_key is None:
            for k in shape_map.keys():
                try:
                    cand = self._parse_shape_key(k)
                except Exception:
                    continue
                if len(cand) == len(target) and cand[0] == target[0] and cand[-1] == target[-1]:
                    best_key = k
                    break
    
        return shape_map.get(best_key) if best_key else None


    def should_fuse(
        self,
        graph: SubGraph,
        input_shapes: Dict[str, Tuple[int, ...]],
        margin: float = 0.0,
    ) -> bool:
        """
        判断是否应融合（静态规则 + profile 决策）
        
        决策条件：unfused_time > fused_time * (1 + margin)
        """
        # 总开关
        if not self.config.enable_fusion:
            return False

        # 节点数检查
        if len(graph.nodes) < self.config.min_nodes_for_fusion:
            if self.config.debug_mode:
                print(f"[Fusion] Skip: node count {len(graph.nodes)} < {self.config.min_nodes_for_fusion}")
            return False

        # 图大小上限
        if len(graph.nodes) > self.config.max_graph_size:
            if self.config.debug_mode:
                print(f"[Fusion] Skip: node count {len(graph.nodes)} > {self.config.max_graph_size}")
            return False

        # 张量大小检查
        for name, shape in input_shapes.items():
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            if num_elements < self.config.min_tensor_elements:
                if self.config.debug_mode:
                    print(f"[Fusion] Skip: tensor '{name}' elements {num_elements} < {self.config.min_tensor_elements}")
                return False

        # 算子类型检查
        supported = self._get_ops()
        for node in graph.nodes:
            if node.op_type not in supported:
                if self.config.debug_mode:
                    print(f"[Fusion] Skip: unsupported op '{node.op_type}'")
                return False

        # Profile 决策
        shape_key = self._pick_profile_shape_key(graph, input_shapes)
        if shape_key is None:
            print("[Fusion][Error] Cannot pick representative shape key")
            return False

        op_key = self._fused_op_key(graph)

        try:
            profile = self._load_profile()
        except Exception as e:
            print(f"[Fusion][Error] Failed to load profile: {e}")
            return False
        
        if self.config.debug_mode:
            print(f"[Fusion] Using profile: {self.profile_path}")

        try:
            unfused_map = profile["unfused"][op_key]
            fused_map = profile["fused"][op_key]
        except Exception as e:
            print(f"[Fusion][Error] Invalid profile structure for op '{op_key}': {e}")
            return False
        
        t_unfused = unfused_map.get(shape_key)
        if t_unfused is None:
            t_unfused = self._lookup_nearest_shape(shape_key, unfused_map)

        t_fused = fused_map.get(shape_key)
        if t_fused is None:
            t_fused = self._lookup_nearest_shape(shape_key, fused_map)

        if t_unfused is None or t_fused is None:
            print(f"[Fusion][Error] Profile missing: op='{op_key}', shape={shape_key}")
            return False

        try:
            t_unfused_f = float(t_unfused)
            t_fused_f = float(t_fused)
            margin_f = float(margin)
        except Exception as e:
            print(f"[Fusion][Error] Invalid profile values: {e}")
            return False

        decision = t_unfused_f > t_fused_f * (1.0 + margin_f)

        if self.config.debug_mode:
            print(
                f"[Fusion] op='{op_key}', shape={shape_key}: "
                f"unfused={t_unfused_f:.4f} fused={t_fused_f:.4f} margin={margin_f:.3f} => "
                f"{'FUSE' if decision else 'SKIP'}"
            )

        return decision

    def get_supported_ops(self) -> Set[str]:
        """返回当前支持融合的算子类型集合"""
        return self._get_ops().copy()
