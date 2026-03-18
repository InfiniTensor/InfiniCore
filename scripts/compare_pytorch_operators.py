#!/usr/bin python3
"""
PyTorch vs InfiniCore 算子对比脚本（简化版）

只统计 torch 和 torch.nn.functional，"""

import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional


@dataclass
class OperatorInfo:
    """算子信息"""
    name: str
    pytorch_equivalent: str = ""
    supported_platforms: List[str] = field(default_factory=list)
    category: str = "other"


# ============================================================================
# 名称归一化
# ============================================================================

def normalize_name(name: str) -> str:
    """归一化算子名称"""
    if not name:
        return name
    # 去掉前导下划线
    while name.startswith('_'):
        name = name[1:]
    # 去掉原地操作后缀
    if name.endswith('_') and len(name) > 1:
        name = name[:-1]
    return name.lower()


# ============================================================================
# LLM 核心算子
# ============================================================================

LLM_CORE_OPS = {
    # 矩阵乘法
    'matmul', 'linear', 'mm', 'bmm', 'addmm', 'gemm',
    # 归一化
    'layer_norm', 'rms_norm', 'batch_norm', 'group_norm',
    # 激活函数
    'relu', 'gelu', 'silu', 'sigmoid', 'tanh', 'softmax', 'log_softmax',
    # Attention
    'scaled_dot_product_attention', 'attention', 'flash_attention',
    'paged_attention', 'kv_caching',
    # 量化
    'quantize', 'dequantize', 'per_channel_quant', 'scaled_mm', 'int_mm',
    # 元素操作
    'add', 'mul', 'sub', 'div', 'pow', 'exp', 'log', 'sqrt',
    # 形状操作
    'reshape', 'transpose', 'permute', 'squeeze', 'unsqueeze', 'view',
    # Embedding
    'embedding', 'rotary_embedding', 'rope',
    # 池化
    'avg_pool', 'max_pool', 'adaptive_avg_pool', 'adaptive_max_pool',
    # 其他
    'concat', 'stack', 'split', 'chunk', 'topk', 'where', 'clip', 'clamp',
}

LLM_CORE_OPS_NORMALIZED = {normalize_name(op) for op in LLM_CORE_OPS}


# ============================================================================
# 名称映射
# ============================================================================

NAME_SYNONYMS = {
    'rms_norm': 'torch.nn.functional.rms_norm',
    'layer_norm': 'torch.nn.functional.layer_norm',
    'rmsnorm': 'torch.nn.functional.rms_norm',
    'layernorm': 'torch.nn.functional.layer_norm',
    'batchnorm': 'torch.nn.functional.batch_norm',
    'groupnorm': 'torch.nn.functional.group_norm',
    'add_rms_norm': 'fused: add + rms_norm',
    'gelu': 'torch.nn.functional.gelu',
    'silu': 'torch.nn.functional.silu',
    'relu': 'torch.nn.functional.relu',
    'softmax': 'torch.nn.functional.softmax',
    'log_softmax': 'torch.nn.functional.log_softmax',
    'logsoftmax': 'torch.nn.functional.log_softmax',
    'causal_softmax': 'torch.nn.functional.softmax (causal)',
    'cross_entropy': 'torch.nn.functional.cross_entropy',
    'embedding': 'torch.nn.functional.embedding',
    'rope': 'rotary position embedding',
    'scaled_dot_product_attention': 'torch.nn.functional.scaled_dot_product_attention',
    'attention': 'torch.nn.functional.scaled_dot_product_attention',
    'paged_attention': 'vLLM paged_attention',
    'flash_attention': 'torch.nn.functional.scaled_dot_product_attention',
    'kv_caching': 'KV cache',
    'swiglu': 'SwiGLU',
    'silu_and_mul': 'fused: silu * mul',
    'gemm': 'torch.matmul',
    'linear': 'torch.nn.functional.linear',
    'conv': 'torch.nn.functional.conv2d',
    'dequantize_awq': 'AWQ dequantization',
    'per_channel_quant_int8': 'torch.quantize_per_channel',
    'asinh': 'torch.asinh',
    'atanh': 'torch.atanh',
    'hardtanh': 'torch.nn.functional.hardtanh',
    'hardswish': 'torch.nn.functional.hardswish',
    'softplus': 'torch.nn.functional.softplus',
    'cdist': 'torch.cdist',
    'addcmul': 'torch.addcmul',
    'clip': 'torch.clamp',
    'reciprocal': 'torch.reciprocal',
    'equal': 'torch.eq',
    'ones': 'torch.ones',
    'zeros': 'torch.zeros',
    'random_sample': 'torch.multinomial',
    'avg_pool1d': 'torch.nn.functional.avg_pool1d',
    'adaptive_max_pool1d': 'torch.nn.functional.adaptive_max_pool1d',
    'topk': 'torch.topk',
    'var': 'torch.var',
    'var_mean': 'torch.var_mean',
    'all': 'torch.all',
    'sum': 'torch.sum',
    'topksoftmax': 'fused: topk + softmax (MoE)',
    'topkrouter': 'MoE top-k router',
    'fmod': 'torch.fmod',
    'rearrange': 'torch.permute / einops.rearrange',
    'scaled_mm': 'torch._int_mm (INT8 matmul)',
    'paged_caching': 'vLLM paged KV cache',
}


# ============================================================================
# 非 PyTorch 算子过滤列表
# ============================================================================

TORCH_NON_OPERATOR_FUNCTIONS = {
    # 子模块
    'nn', 'optim', 'autograd', 'cuda', 'backends', 'distributed', 'jit', 'onnx',
    'utils', 'data', 'vision', 'audio', 'text', 'quantization', 'amp', 'fx',
    # 工厂函数
    'tensor', 'as_tensor', 'from_numpy', 'empty', 'zeros', 'ones', 'rand', 'randn',
    'arange', 'linspace', 'eye', 'diag',
    # 持久化
    'save', 'load',
    # 梯度控制
    'no_grad', 'enable_grad', 'set_grad_enabled',
    # 设备管理
    'cuda', 'cpu', 'device', 'to',
    # 配置
    'set_printoptions', 'get_default_dtype', 'set_default_dtype',
    # 工具
    'clone', 'copy_', 'detach', 'requires_grad_', 'backward',
    # 类型
    'float', 'double', 'half', 'int', 'long', 'short', 'bool', 'byte',
    'Tensor', 'Size', 'dtype', 'Device',
}


class PyTorchCollector:
    """收集 PyTorch 算子（只收集 torch 和 nn.functional）"""

    def __init__(self):
        self.nn_functional: Set[str] = set()
        self.torch_functions: Set[str] = set()

    def collect(self):
        """收集算子"""
        try:
            import torch
            import torch.nn.functional as F
            import inspect

            # 收集 nn.functional
            for name in dir(F):
                if not name.startswith('_'):
                    obj = getattr(F, name, None)
                    if callable(obj) and not inspect.ismodule(obj):
                        self.nn_functional.add(name)

            # 收集 torch 函数
            for name in dir(torch):
                if not name.startswith('_') and name not in TORCH_NON_OPERATOR_FUNCTIONS:
                    obj = getattr(torch, name, None)
                    if callable(obj) and not isinstance(obj, type):
                        obj_type = str(type(obj))
                        if 'module' not in obj_type.lower():
                            self.torch_functions.add(name)

            print(f"   nn.functional: {len(self.nn_functional)}")
            print(f"   torch:          {len(self.torch_functions)}")

        except ImportError:
            print("Warning: PyTorch not installed")

    def get_summary(self) -> dict:
        """获取统计摘要"""
        nn = self.nn_functional
        torch_funcs = self.torch_functions

        overlap = nn & torch_funcs
        union = nn | torch_funcs

        nn_only = nn - torch_funcs
        torch_only = torch_funcs - nn

        # 归一化
        nn_norm = {normalize_name(n) for n in nn}
        torch_norm = {normalize_name(n) for n in torch_funcs}
        union_norm = nn_norm | torch_norm

        # LLM 核心算子
        all_norm = nn_norm | torch_norm
        core_found = LLM_CORE_OPS_NORMALIZED & all_norm
        core_missing = LLM_CORE_OPS_NORMALIZED - all_norm
        core_coverage = len(core_found) / len(LLM_CORE_OPS_NORMALIZED) * 100 if LLM_CORE_OPS_NORMALIZED else 0

        return {
            'nn_functional_count': len(nn),
            'torch_count': len(torch_funcs),
            'overlap': len(overlap),
            'union': len(union),
            'nn_only_count': len(nn_only),
            'torch_only_count': len(torch_only),
            'nn_only_list': sorted(nn_only),
            'torch_only_list': sorted(torch_only),
            'common_list': sorted(overlap),
            'normalized_union': len(union_norm),
            'llm_core_total': len(LLM_CORE_OPS_NORMALIZED),
            'llm_core_found': len(core_found),
            'llm_core_missing': len(core_missing),
            'llm_core_coverage': core_coverage,
            'llm_core_found_list': sorted(core_found),
            'llm_core_missing_list': sorted(core_missing),
        }


class InfiniCoreCollector:
    """从 InfiniCore 源码收集算子"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.ops: Dict[str, OperatorInfo] = {}
        self.pytorch_mappings: Dict[str, str] = {}

    def _categorize(self, op_name: str) -> str:
        """对算子进行分类"""
        categories = {
            "activation": {
                "relu", "sigmoid", "tanh", "gelu", "silu", "softplus",
                "hardtanh", "hardswish", "atanh", "asinh", "leaky_relu",
                "elu", "prelu", "mish", "glu"
            },
            "normalization": {
                "layer_norm", "rms_norm", "add_rms_norm", "lp_norm",
                "batch_norm", "group_norm", "instance_norm"
            },
            "attention": {
                "attention", "causal_softmax", "flash_attention",
                "paged_attention", "paged_attention_prefill", "kv_caching",
                "paged_caching", "scaled_dot_product_attention"
            },
            "linear": {"gemm", "linear", "matmul", "mm", "bmm", "addmm"},
            "convolution": {"conv", "conv1d", "conv2d", "conv3d", "conv_transpose"},
            "pooling": {
                "avg_pool1d", "avg_pool2d", "max_pool1d", "max_pool2d",
                "adaptive_max_pool1d", "adaptive_avg_pool1d"
            },
            "softmax": {"softmax", "logsoftmax", "topksoftmax"},
            "math": {
                "add", "sub", "mul", "div", "reciprocal", "fmod", "addcmul",
                "clip", "clamp", "pow", "sqrt", "exp", "log"
            },
            "reduction": {
                "sum", "mean", "var", "std", "prod", "max", "min",
                "argmax", "argmin", "all", "any", "topk", "var_mean"
            },
            "embedding": {"embedding", "one_hot"},
            "loss": {
                "cross_entropy", "binary_cross_entropy", "mse_loss",
                "nll_loss", "kl_div", "binary_cross_entropy_with_logits"
            },
            "distance": {"cdist", "pdist", "cosine_similarity"},
            "comparison": {"equal", "eq", "ne", "gt", "lt", "ge", "le"},
            "tensor_ops": {
                "rearrange", "reshape", "view", "transpose", "permute",
                "squeeze", "unsqueeze", "cat", "stack", "gather", "scatter"
            },
            "positional_encoding": {"rope", "rotary_embedding"},
            "fused_ops": {
                "swiglu", "silu_and_mul", "topkrouter", "geglu",
                "add_rms_norm", "fused_moe"
            },
            "quantization": {
                "quantize", "dequantize", "per_channel_quant_int8",
                "scaled_mm", "dequantize_awq"
            },
            "random": {
                "random_sample", "rand", "randn", "randint", "multinomial"
            },
            "initialization": {"ones", "zeros", "full", "empty", "arange"},
        }

        for category, ops in categories.items():
            if op_name in ops:
                return category

        return "other"

    def collect(self) -> Dict[str, OperatorInfo]:
        """收集算子"""
        self._extract_pytorch_mappings_from_tests()
        self._collect_from_ops_directory()
        return self.ops

    def _extract_pytorch_mappings_from_tests(self):
        """从测试文件提取 PyTorch 映射"""
        test_dirs = [
            self.repo_path / "test" / "infiniop",
            self.repo_path / "test" / "python",
        ]

        patterns = [
            r'torch\.(\w+)\s*\(',
            r'torch\.nn\.functional\.(\w+)\s*\(',
            r'\bF\.(\w+)\s*\(',
        ]

        for test_dir in test_dirs:
            if not test_dir.exists():
                continue
            for py_file in test_dir.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    for pattern in patterns:
                        matches = re.findall(pattern, content)
                        for match in matches:
                            if match not in ('Callable', 'Optional', 'List', 'Tuple', 'Tensor'):
                                if 'nn.functional' in pattern or pattern.startswith(r'\bF\.'):
                                    self.pytorch_mappings[match] = f"torch.nn.functional.{match}"
                                else:
                                    self.pytorch_mappings[match] = f"torch.{match}"
                except Exception:
                    pass

    def _detect_platforms(self, op_dir: Path) -> List[str]:
        """检测算子支持的平台

        通过检查 operator.cc 中的 ENABLE_XXX_API 和 INFINI_DEVICE_XXX 宏来检测平台
        共11个平台：CPU, NVIDIA, ILUVATAR, ALI, QY, HYGON, KUNLUN, CAMBRICON, ASCEND, METAX, MOORE
        """
        platforms = set()

        # 目录名到平台名的映射（用于检测子目录实现）
        dir_to_platform = {
            'cpu': 'CPU',
            'cuda': 'NVIDIA',
            'nvidia': 'NVIDIA',
            'ascend': 'Ascend',
            'bang': 'Cambricon',
            'kunlun': 'Kunlun',
            'metax': 'MetaX',
            'moore': 'Moore',
        }

        # 检查子目录（部分算子有平台特定实现目录）
        for subdir in op_dir.iterdir():
            if subdir.is_dir():
                plat_name = dir_to_platform.get(subdir.name.lower())
                if plat_name:
                    platforms.add(plat_name)

        # 从 operator.cc 中检测 ENABLE_XXX_API 宏
        # 这是更准确的方式，直接反映编译时的平台支持
        device_to_platform = {
            'CPU': 'CPU',
            'NVIDIA': 'NVIDIA',
            'ILUVATAR': 'Iluvatar',
            'ALI': 'Aliyun',
            'QY': 'Qiyuan',
            'HYGON': 'Hygon',
            'KUNLUN': 'Kunlun',
            'CAMBRICON': 'Cambricon',
            'ASCEND': 'Ascend',
            'METAX': 'MetaX',
            'MOORE': 'Moore',
        }

        operator_cc = op_dir / "operator.cc"
        if operator_cc.exists():
            try:
                content = operator_cc.read_text(encoding='utf-8')
                for device_type, plat_name in device_to_platform.items():
                    # 检查 ENABLE_XXX_API 或 INFINI_DEVICE_XXX
                    if f'ENABLE_{device_type}_API' in content or f'INFINI_DEVICE_{device_type}' in content:
                        platforms.add(plat_name)
                # 如果没有检测到任何平台宏，但有 __export 函数，说明是动态支持的组合算子
                # 这类算子支持所有平台（平台分配逻辑在底层算子）
                if not platforms and '__export' in content:
                    platforms = set(device_to_platform.values())
            except Exception:
                pass

        return sorted(platforms) if platforms else ['Unknown']

    def _collect_from_ops_directory(self):
        """从 ops 目录收集算子"""
        ops_dir = self.repo_path / "src" / "infiniop" / "ops"
        if not ops_dir.exists():
            print(f"Warning: {ops_dir} not found")
            return

        for operator_cc in ops_dir.rglob("operator.cc"):
            op_dir = operator_cc.parent
            rel_path = op_dir.relative_to(ops_dir)

            if len(rel_path.parts) > 1:
                op_name = rel_path.parts[-1]
            else:
                op_name = rel_path.name

            if op_name in self.ops:
                continue

            platforms = self._detect_platforms(op_dir)
            pytorch_eq = self.pytorch_mappings.get(op_name, "") or NAME_SYNONYMS.get(op_name, "")

            self.ops[op_name] = OperatorInfo(
                name=op_name,
                pytorch_equivalent=pytorch_eq,
                supported_platforms=platforms,
                category=self._categorize(op_name),
            )


def generate_report(pytorch: PyTorchCollector, infinicore: Dict[str, OperatorInfo], output_path: str):
    """生成对比报告"""
    summary = pytorch.get_summary()

    report = []
    report.append("# PyTorch vs InfiniCore 算子对比报告\n")
    report.append(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # PyTorch 统计
    report.append("## 📊 PyTorch 算子统计\n")
    report.append("> 只统计 `torch` 和 `torch.nn.functional`（ATen 不对用户暴露，Tensor 方法大部分在 torch 有对应）\n")
    report.append("")
    report.append("| 模块 | 数量 | 说明 |")
    report.append("|------|------|------|")
    report.append(f"| torch.nn.functional | {summary['nn_functional_count']} | 神经网络专用函数（最规范） |")
    report.append(f"| torch | {summary['torch_count']} | 通用张量操作 |")
    report.append(f"| **两者重叠** | {summary['overlap']} | 同时存在于两个模块 |")
    report.append(f"| **并集（去重）** | **{summary['union']}** | 合并去重 |")
    report.append("")

    # 重叠分析
    report.append("### 重叠分析\n")
    report.append("| 分类 | 数量 |")
    report.append("|------|------|")
    report.append(f"| nn.functional 独有 | {summary['nn_only_count']} |")
    report.append(f"| torch 独有 | {summary['torch_only_count']} |")
    report.append(f"| 两者共有 | {len(summary['common_list'])} |")
    report.append("")

    # LLM 核心算子
    report.append("### 🎯 LLM 核心算子覆盖\n")
    report.append(f"> 定义了 {summary['llm_core_total']} 个 LLM 推理最关键的算子\n")
    report.append("")
    report.append("| 指标 | 数值 |")
    report.append("|------|------|")
    report.append(f"| PyTorch 覆盖 | {summary['llm_core_found']}/{summary['llm_core_total']} ({summary['llm_core_coverage']:.1f}%) |")
    report.append(f"| InfiniCore 覆盖 | 见下方统计 |")
    report.append("")

    if summary['llm_core_found_list']:
        report.append(f"**PyTorch 已覆盖**: {', '.join(summary['llm_core_found_list'])}")
    if summary['llm_core_missing_list']:
        report.append(f"**PyTorch 未覆盖**: {', '.join(summary['llm_core_missing_list'])}")
    report.append("")

    # InfiniCore 统计
    report.append("## 📊 InfiniCore 算子统计\n")
    report.append(f"| 指标 | 数量 |")
    report.append("|------|------|")
    report.append(f"| InfiniCore 算子总数 | {len(infinicore)} |")

    def is_pytorch_mapping(m: str) -> bool:
        return m and m.startswith('torch.')

    has_mapping = sum(1 for op in infinicore.values() if is_pytorch_mapping(op.pytorch_equivalent))
    report.append(f"| 有 PyTorch 对应 | {has_mapping} |")
    report.append(f"| InfiniCore 特有 | {len(infinicore) - has_mapping} |")

    # InfiniCore LLM 核心算子
    inf_names_norm = {normalize_name(n) for n in infinicore.keys()}
    for op in infinicore.values():
        if op.pytorch_equivalent and op.pytorch_equivalent.startswith('torch.'):
                parts = op.pytorch_equivalent.split('.')
                if len(parts) >= 2:
                    inf_names_norm.add(normalize_name(parts[-1]))

    inf_core_found = LLM_CORE_OPS_NORMALIZED & inf_names_norm
    inf_core_missing = LLM_CORE_OPS_NORMALIZED - inf_names_norm
    inf_core_coverage = len(inf_core_found) / len(LLM_CORE_OPS_NORMALIZED) * 100 if LLM_CORE_OPS_NORMALIZED else 0

    report.append(f"| **LLM 核心算子覆盖** | **{len(inf_core_found)}/{len(LLM_CORE_OPS_NORMALIZED)} ({inf_core_coverage:.1f}%)** |")
    report.append("")

    # 列出具体算子
    pytorch_mapped_ops = [op.name for op in infinicore.values() if is_pytorch_mapping(op.pytorch_equivalent)]
    unique_ops = [op.name for op in infinicore.values() if not is_pytorch_mapping(op.pytorch_equivalent)]
    report.append(f"**有 PyTorch 对应 ({len(pytorch_mapped_ops)})**: {', '.join(f'`{op}`' for op in sorted(pytorch_mapped_ops))}")
    report.append("")
    report.append(f"**InfiniCore 特有 ({len(unique_ops)})**: {', '.join(f'`{op}`' for op in sorted(unique_ops))}")
    report.append("")

    report.append("### 🎯 InfiniCore LLM 核心算子详情\n")
    if inf_core_found:
        report.append(f"**✅ 已支持 ({len(inf_core_found)})**: {', '.join(sorted(inf_core_found))}")
        report.append("")
    if inf_core_missing:
        report.append(f"**❌ 未支持 ({len(inf_core_missing)})**: {', '.join(sorted(inf_core_missing))}")
        report.append("")
    report.append("> 支持这些核心算子即可覆盖大部分 LLM 推理场景\n")

    # 模块对应分析
    report.append("## 📊 PyTorch 模块对应分析\n")

    pytorch_modules = defaultdict(list)
    for op_info in infinicore.values():
        mapping = op_info.pytorch_equivalent
        if is_pytorch_mapping(mapping):
            parts = mapping.split('.')
            if len(parts) >= 3 and parts[1] == 'nn' and parts[2] == 'functional':
                module = 'torch.nn.functional'
            else:
                module = 'torch'
            pytorch_modules[module].append(op_info.name)

    module_counts = {
        'torch': summary['torch_count'],
        'torch.nn.functional': summary['nn_functional_count'],
    }

    def get_level(pct):
        if pct >= 50:
                    return "🟢 高"
        elif pct >= 20:
                    return "🟡 中"
        else:
                    return "🔴 低"

    report.append("| PyTorch 模块 | 模块算子数 | 对应算子数 | 覆盖率 | 覆盖程度 |")
    report.append("|-------------|-----------|-----------|-------|---------|")

    for module in ['torch', 'torch.nn.functional']:
        ops = pytorch_modules.get(module, [])
        total = module_counts.get(module, 0)
        pct = len(ops) / total * 100 if total > 0 else 0
        level = get_level(pct)
        report.append(f"| `{module}` | {total} | {len(ops)} | {pct:.1f}% | {level} |")

    report.append("")

    # ============================================================
    # 平台支持统计
    # ============================================================
    report.append("## 📊 平台支持统计\n")

    # 统计每个平台支持多少算子
    platform_op_count = defaultdict(int)
    for op_info in infinicore.values():
        for platform in op_info.supported_platforms:
            platform_op_count[platform] += 1

    total_ops = len(infinicore)

    report.append("### 各平台算子覆盖\n")
    report.append("| 平台 | 支持算子数 | 覆盖率 |")
    report.append("|------|-----------|--------|")
    for platform, count in sorted(platform_op_count.items(), key=lambda x: -x[1]):
        coverage = count / total_ops * 100
        bar = "█" * int(coverage / 5) + "░" * (20 - int(coverage / 5))
        report.append(f"| {platform} | {count} | {coverage:.1f}% {bar} |")
    report.append("")

    # 统计算子支持的平台数分布
    report.append("### 算子跨平台支持分布\n")
    report.append("| 支持平台数 | 算子数量 | 占比 | 算子列表 |")
    report.append("|-----------|---------|------|----------|")

    platform_count_dist = defaultdict(list)
    for op_info in infinicore.values():
        count = len(op_info.supported_platforms)
        platform_count_dist[count].append(op_info.name)

    for count in sorted(platform_count_dist.keys(), reverse=True):
        ops = platform_count_dist[count]
        pct = len(ops) / total_ops * 100
        # 列出所有算子
        ops_list = ", ".join(f"`{op}`" for op in sorted(ops))
        report.append(f"| {count} 个平台 | {len(ops)} | {pct:.1f}% | {ops_list} |")
    report.append("")

    # 只支持少量平台的算子（需要扩展）
    low_support_ops = [(op.name, op.supported_platforms)
                       for op in infinicore.values()
                       if len(op.supported_platforms) <= 3]

    # ============================================================
    # 数据驱动的开发建议
    # ============================================================
    report.append("## 💡 开发建议\n")
    report.append("> 以下建议基于实际数据分析生成\n")

    # 建议1: 扩展平台支持
    report.append("### 1. 扩展平台支持（优先级高）\n")
    if low_support_ops:
        report.append("这些算子只支持少量平台，建议扩展到更多平台：\n")
        for op_name, platforms in sorted(low_support_ops, key=lambda x: len(x[1]))[:10]:
            report.append(f"- **`{op_name}`** - 当前: {len(platforms)} 平台 ({', '.join(platforms) if platforms else '无'})")
    else:
        report.append("✅ 所有算子都支持 4 个以上平台！\n")
    report.append("")

    # 建议2: LLM 推理常用算子覆盖
    report.append("### 2. LLM 推理常用算子覆盖\n")
    llm_priority = {
        "scaled_dot_product_attention": "注意力机制",
        "layer_norm": "层归一化",
        "rms_norm": "RMS归一化",
        "silu": "SiLU激活",
        "gelu": "GELU激活",
        "softmax": "Softmax",
        "rope": "旋转位置编码",
        "kv_caching": "KV缓存",
        "gemm": "矩阵乘法",
        "embedding": "词嵌入",
    }

    # 构建小写名称映射
    inf_names_lower = {name.lower().replace('_', ''): (name, info)
                       for name, info in infinicore.items()}

    def find_infinicore_op_for_pytorch(pytorch_op: str):
        """根据 PyTorch 算子名找到对应的 InfiniCore 算子"""
        op_norm = pytorch_op.lower().replace('_', '')

        # 方法1: 名称直接匹配
        if op_norm in inf_names_lower:
            return inf_names_lower[op_norm]

        # 方法2: 在 PyTorch 等价映射中查找
        for inf_name, info in infinicore.items():
            pytorch_eq = info.pytorch_equivalent
            if pytorch_eq and pytorch_op.lower().replace('_', '') in pytorch_eq.lower().replace('_', '').replace('.', ''):
                return (inf_name, info)

        return None

    report.append("| 算子 | 用途 | 状态 | InfiniCore 对应 | 支持平台数 |")
    report.append("|------|------|------|-----------------|----------|")
    for op, desc in llm_priority.items():
        result = find_infinicore_op_for_pytorch(op)
        if result:
            status = "✅"
            inf_name, info = result
            platform_count = len(info.supported_platforms)
            infinicore_name = f"`{inf_name}`"
        else:
            status = "❌"
            platform_count = "-"
            infinicore_name = "-"
        report.append(f"| `{op}` | {desc} | {status} | {infinicore_name} | {platform_count} |")
    report.append("")

    # 建议3: 平台覆盖均衡性
    report.append("### 3. 平台覆盖均衡性\n")
    avg_coverage = sum(platform_op_count.values()) / len(platform_op_count) if platform_op_count else 0
    below_avg = [(p, c) for p, c in platform_op_count.items() if c < avg_coverage]
    if below_avg:
        report.append(f"平均每个平台支持 {avg_coverage:.1f} 个算子。以下平台低于平均值：\n")
        for platform, count in sorted(below_avg, key=lambda x: x[1]):
            report.append(f"- **{platform}**: {count} 个算子 (差 {int(avg_coverage - count)} 个)")
    else:
        report.append("✅ 所有平台覆盖率都在平均值以上！\n")
    report.append("")

    # ============================================================
    # InfiniCore 算子详情
    # ============================================================
    report.append("## 🔧 InfiniCore 支持的算子\n")

    # 按分类组织
    by_category = defaultdict(list)
    for name, info in infinicore.items():
        by_category[info.category].append(info)

    for category in sorted(by_category.keys()):
        ops = by_category[category]
        report.append(f"### {category.upper()}\n")
        report.append("| InfiniCore 算子 | PyTorch 等价 | 平台数 | 支持平台 |")
        report.append("|-----------------|--------------|--------|----------|")

        for op in sorted(ops, key=lambda x: x.name):
            platform_count = len(op.supported_platforms)
            platforms_str = ", ".join(op.supported_platforms) if op.supported_platforms else "CPU"
            pytorch_eq = op.pytorch_equivalent if op.pytorch_equivalent else "-"
            report.append(f"| `{op.name}` | {pytorch_eq} | {platform_count} | {platforms_str} |")

        report.append("")

    # ============================================================
    # InfiniCore 特有算子（没有标准 PyTorch 对应的算子）
    # ============================================================
    def is_real_pytorch_op(mapping: str) -> bool:
        """判断是否是真正的 PyTorch 函数映射"""
        if not mapping:
            return False
        return mapping.startswith('torch.')

    unique_ops = [op for op in infinicore.values() if not is_real_pytorch_op(op.pytorch_equivalent)]
    if unique_ops:
        report.append("## ⭐ InfiniCore 特有算子\n")
        report.append("这些算子没有标准的 PyTorch 对应函数：\n")
        report.append("| 算子 | 分类 | 当前描述 | 平台数 | 支持平台 |")
        report.append("|------|------|----------|--------|----------|")
        for op in sorted(unique_ops, key=lambda x: x.name):
            platform_count = len(op.supported_platforms)
            platforms_str = ", ".join(op.supported_platforms) if op.supported_platforms else "CPU"
            desc = op.pytorch_equivalent if op.pytorch_equivalent else "-"
            report.append(f"| `{op.name}` | {op.category} | {desc} | {platform_count} | {platforms_str} |")
        report.append("")

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"   报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch vs InfiniCore 算子对比（简化版）')
    parser.add_argument('--repo', default='.', help='InfiniCore 仓库路径')
    parser.add_argument('--output', default='operator_comparison_report.md', help='输出报告路径')
    args = parser.parse_args()

    print("=" * 60)
    print("PyTorch vs InfiniCore 算子对比（简化版）")
    print("=" * 60)

    print("\n[1/3] 收集 PyTorch 算子...")
    pytorch = PyTorchCollector()
    pytorch.collect()

    print("\n[2/3] 收集 InfiniCore 算子...")
    infinicore = InfiniCoreCollector(args.repo).collect()
    print(f"   找到 {len(infinicore)} 个算子")

    print("\n[3/3] 生成报告...")
    output_path = os.path.join(args.repo, args.output)
    generate_report(pytorch, infinicore, output_path)

    summary = pytorch.get_summary()
    print("\n" + "=" * 60)
    print("📊 对比摘要")
    print("=" * 60)
    print(f"PyTorch 算子（torch + nn.functional）: {summary['union']}")
    print(f"InfiniCore 算子:                       {len(infinicore)}")
    print(f"nn.functional 覆盖率:            {summary['llm_core_coverage']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
