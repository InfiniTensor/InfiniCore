"""
统一测试所有 Unary 算子

这个文件包含所有 unary 算子的测试，方便统一管理和运行。
可以通过命令行参数选择运行哪些算子，或者运行所有算子。

使用方法:
    # 运行所有 unary 算子测试
    python test_all_unary_ops.py
    
    # 只运行 abs 和 log 算子
    python test_all_unary_ops.py --ops abs log
    
    # 运行特定设备上的测试
    python test_all_unary_ops.py --cpu --nvidia
"""

import torch
import argparse
from libinfiniop import InfiniDtype
from libinfiniop.unary_test_base import UnaryTestBase


# ==============================================================================
# 所有 Unary 算子的测试类定义
# ==============================================================================

class AbsTest(UnaryTestBase):
    OP_NAME = "Abs"
    OP_NAME_LOWER = "abs"
    
    @staticmethod
    def torch_op(x):
        return torch.abs(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }


class AcosTest(UnaryTestBase):
    OP_NAME = "Acos"
    OP_NAME_LOWER = "acos"
    
    @staticmethod
    def torch_op(x):
        return torch.acos(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # acos domain is [-1, 1]
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class AcoshTest(UnaryTestBase):
    OP_NAME = "Acosh"
    OP_NAME_LOWER = "acosh"
    
    @staticmethod
    def torch_op(x):
        return torch.acosh(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # acosh domain is [1, +∞)
        return torch.rand(shape, dtype=dtype, device=device) * 10 + 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class AsinTest(UnaryTestBase):
    OP_NAME = "Asin"
    OP_NAME_LOWER = "asin"
    
    @staticmethod
    def torch_op(x):
        return torch.asin(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # asin domain is [-1, 1]
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class AsinhTest(UnaryTestBase):
    OP_NAME = "Asinh"
    OP_NAME_LOWER = "asinh"
    
    @staticmethod
    def torch_op(x):
        return torch.asinh(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class AtanTest(UnaryTestBase):
    OP_NAME = "Atan"
    OP_NAME_LOWER = "atan"
    
    @staticmethod
    def torch_op(x):
        return torch.atan(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class AtanhTest(UnaryTestBase):
    OP_NAME = "Atanh"
    OP_NAME_LOWER = "atanh"
    
    @staticmethod
    def torch_op(x):
        return torch.atanh(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # atanh domain is (-1, 1)
        return torch.rand(shape, dtype=dtype, device=device) * 1.8 - 0.9
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class CeilTest(UnaryTestBase):
    OP_NAME = "Ceil"
    OP_NAME_LOWER = "ceil"
    
    @staticmethod
    def torch_op(x):
        return torch.ceil(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 10 - 5
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }


class SinTest(UnaryTestBase):
    OP_NAME = "Sin"
    OP_NAME_LOWER = "sin"
    
    @staticmethod
    def torch_op(x):
        return torch.sin(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # Generate test tensors with values in range [-200, -100) for sin operation
        # sin domain is (-∞, +∞), so we use range [-200, -100)
        return torch.rand(shape, dtype=dtype, device=device) * 100 - 200
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-4, "rtol": 1e-2},
        InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-2},
    }
    
    EQUAL_NAN = True


class CosTest(UnaryTestBase):
    OP_NAME = "Cos"
    OP_NAME_LOWER = "cos"
    
    @staticmethod
    def torch_op(x):
        return torch.cos(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # Generate test tensors with values in range [-200, -100) for cos operation
        # cos domain is (-∞, +∞), so we use range [-200, -100)
        return torch.rand(shape, dtype=dtype, device=device) * 100 - 200
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-4, "rtol": 1e-2},
        InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-2},
    }
    
    EQUAL_NAN = True


class CoshTest(UnaryTestBase):
    OP_NAME = "Cosh"
    OP_NAME_LOWER = "cosh"
    
    @staticmethod
    def torch_op(x):
        return torch.cosh(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class ErfTest(UnaryTestBase):
    OP_NAME = "Erf"
    OP_NAME_LOWER = "erf"
    
    @staticmethod
    def torch_op(x):
        return torch.erf(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class FloorTest(UnaryTestBase):
    OP_NAME = "Floor"
    OP_NAME_LOWER = "floor"
    
    @staticmethod
    def torch_op(x):
        return torch.floor(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 10 - 5
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class LogTest(UnaryTestBase):
    OP_NAME = "Log"
    OP_NAME_LOWER = "log"
    
    @staticmethod
    def torch_op(x):
        return torch.log(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # log domain is (0, +∞), so we use range [0.1, 1.1)
        return torch.rand(shape, dtype=dtype, device=device) + 0.1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-7, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-3},
    }
    
    EQUAL_NAN = True


class Log2Test(UnaryTestBase):
    OP_NAME = "Log2"
    OP_NAME_LOWER = "log2"
    
    @staticmethod
    def torch_op(x):
        return torch.log2(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # log2 domain is (0, +∞), so we use range [0.1, 1.1)
        return torch.rand(shape, dtype=dtype, device=device) + 0.1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-7, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-3},
        InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
    }
    
    # Support BF16
    TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]
    
    EQUAL_NAN = True


class Log10Test(UnaryTestBase):
    OP_NAME = "Log10"
    OP_NAME_LOWER = "log10"
    
    @staticmethod
    def torch_op(x):
        return torch.log10(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # log10 domain is (0, +∞), so we use range [0.1, 1.1)
        return torch.rand(shape, dtype=dtype, device=device) + 0.1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-7, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-3},
        InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
    }
    
    # Support BF16
    TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]
    
    EQUAL_NAN = True


class Log1pTest(UnaryTestBase):
    OP_NAME = "Log1p"
    OP_NAME_LOWER = "log1p"
    
    @staticmethod
    def torch_op(x):
        return torch.log1p(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # log1p domain is (-1, +∞), so we use range [-0.9, 1.1)
        # Include values close to zero to test numerical stability
        x = torch.rand(shape, dtype=dtype, device=device) * 2 - 0.9
        return x
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }


class NegTest(UnaryTestBase):
    OP_NAME = "Neg"
    OP_NAME_LOWER = "neg"
    
    @staticmethod
    def torch_op(x):
        return torch.neg(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class ReciprocalTest(UnaryTestBase):
    OP_NAME = "Reciprocal"
    OP_NAME_LOWER = "reciprocal"
    
    @staticmethod
    def torch_op(x):
        return torch.reciprocal(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # Avoid zeros
        return torch.rand(shape, dtype=dtype, device=device) * 2 + 0.1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class RoundTest(UnaryTestBase):
    OP_NAME = "Round"
    OP_NAME_LOWER = "round"
    
    @staticmethod
    def torch_op(x):
        return torch.round(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 10 - 5
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class SignTest(UnaryTestBase):
    OP_NAME = "Sign"
    OP_NAME_LOWER = "sign"
    
    @staticmethod
    def torch_op(x):
        return torch.sign(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class SinhTest(UnaryTestBase):
    OP_NAME = "Sinh"
    OP_NAME_LOWER = "sinh"
    
    @staticmethod
    def torch_op(x):
        return torch.sinh(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class SqrtTest(UnaryTestBase):
    OP_NAME = "Sqrt"
    OP_NAME_LOWER = "sqrt"
    
    @staticmethod
    def torch_op(x):
        return torch.sqrt(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # sqrt domain is [0, +∞)
        return torch.rand(shape, dtype=dtype, device=device) * 100
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 0, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    }
    
    EQUAL_NAN = True


class SquareTest(UnaryTestBase):
    OP_NAME = "Square"
    OP_NAME_LOWER = "square"
    
    @staticmethod
    def torch_op(x):
        return torch.square(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 10 - 5
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class RsqrtTest(UnaryTestBase):
    OP_NAME = "Rsqrt"
    OP_NAME_LOWER = "rsqrt"
    
    @staticmethod
    def torch_op(x):
        return torch.rsqrt(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # rsqrt domain is (0, +∞), avoid zero
        return torch.rand(shape, dtype=dtype, device=device) * 100 + 1e-6
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 2e-3},
        InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    }
    
    EQUAL_NAN = True


class TanTest(UnaryTestBase):
    OP_NAME = "Tan"
    OP_NAME_LOWER = "tan"
    
    @staticmethod
    def torch_op(x):
        return torch.tan(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class ExpTest(UnaryTestBase):
    OP_NAME = "Exp"
    OP_NAME_LOWER = "exp"
    
    @staticmethod
    def torch_op(x):
        return torch.exp(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
        InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
    }
    
    # Support BF16
    TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]


class Exp2Test(UnaryTestBase):
    OP_NAME = "Exp2"
    OP_NAME_LOWER = "exp2"
    
    @staticmethod
    def torch_op(x):
        return torch.exp2(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # Keep input in reasonable range to avoid overflow
        return torch.rand(shape, dtype=dtype, device=device) * 4 - 2
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
        InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
    }
    
    # Support BF16
    TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]


class HardswishTest(UnaryTestBase):
    OP_NAME = "Hardswish"
    OP_NAME_LOWER = "hardswish"
    
    @staticmethod
    def torch_op(x):
        return (x * torch.clamp(x + 3, min=0, max=6) / 6).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
        InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
    }
    
    # Support BF16
    TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]


class IsNanTest(UnaryTestBase):
    OP_NAME = "IsNan"
    OP_NAME_LOWER = "isnan"
    
    @staticmethod
    def torch_op(x):
        return torch.isnan(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # Generate a mix of normal values and NaN values
        x = torch.rand(shape, dtype=dtype, device=device) * 10 - 5
        # Set some values to NaN
        nan_mask = torch.rand(shape, device=device) < 0.3
        x[nan_mask] = float('nan')
        return x
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 0, "rtol": 0},
        InfiniDtype.F32: {"atol": 0, "rtol": 0},
    }
    
    EQUAL_NAN = False  # For isnan, we want exact match (0 or 1)


class IsInfTest(UnaryTestBase):
    OP_NAME = "IsInf"
    OP_NAME_LOWER = "isinf"
    
    @staticmethod
    def torch_op(x):
        return torch.isinf(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # Generate a mix of normal values and Inf values
        x = torch.rand(shape, dtype=dtype, device=device) * 10 - 5
        # Set some values to Inf
        inf_mask = torch.rand(shape, device=device) < 0.3
        x[inf_mask] = float('inf')
        # Set some to -Inf
        neg_inf_mask = torch.rand(shape, device=device) < 0.15
        x[neg_inf_mask] = float('-inf')
        return x
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 0, "rtol": 0},
        InfiniDtype.F32: {"atol": 0, "rtol": 0},
    }
    
    EQUAL_NAN = False  # For isinf, we want exact match (0 or 1)


class IsFiniteTest(UnaryTestBase):
    OP_NAME = "IsFinite"
    OP_NAME_LOWER = "isfinite"
    
    @staticmethod
    def torch_op(x):
        return torch.isfinite(x).to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # Generate a mix of normal values, NaN, and Inf values
        x = torch.rand(shape, dtype=dtype, device=device) * 10 - 5
        # Set some values to NaN
        nan_mask = torch.rand(shape, device=device) < 0.2
        x[nan_mask] = float('nan')
        # Set some values to Inf
        inf_mask = torch.rand(shape, device=device) < 0.2
        x[inf_mask] = float('inf')
        return x
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 0, "rtol": 0},
        InfiniDtype.F32: {"atol": 0, "rtol": 0},
    }
    
    EQUAL_NAN = False  # For isfinite, we want exact match (0 or 1)


class SincTest(UnaryTestBase):
    OP_NAME = "Sinc"
    OP_NAME_LOWER = "sinc"
    
    @staticmethod
    def torch_op(x):
        # PyTorch doesn't have sinc, so we implement it manually
        # sinc(x) = sin(x) / x, sinc(0) = 1
        result = torch.sin(x) / x
        result[x == 0] = 1.0
        return result.to(x.dtype)
    
    @staticmethod
    def generate_input(shape, dtype, device):
        # Generate values around zero and some larger values
        # Include zero to test the special case
        x = torch.rand(shape, dtype=dtype, device=device) * 10 - 5
        # Set some values to exactly zero
        zero_mask = torch.rand(shape, device=device) < 0.1
        x[zero_mask] = 0.0
        return x
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-4},  # sinc can have larger errors near zero
    }
    
    EQUAL_NAN = True


# ==============================================================================
# 算子注册表
# ==============================================================================

# 所有 unary 算子的测试类映射
UNARY_OP_TESTS = {
    "abs": AbsTest,
    "acos": AcosTest,
    "acosh": AcoshTest,
    "asin": AsinTest,
    "asinh": AsinhTest,
    "atan": AtanTest,
    "atanh": AtanhTest,
    "ceil": CeilTest,
    "cos": CosTest,
    "cosh": CoshTest,
    "erf": ErfTest,
    "floor": FloorTest,
    "log": LogTest,
    "log2": Log2Test,
    "log10": Log10Test,
    "log1p": Log1pTest,
    "neg": NegTest,
    "reciprocal": ReciprocalTest,
    "round": RoundTest,
    "sign": SignTest,
    "sin": SinTest,
    "sinh": SinhTest,
    "sqrt": SqrtTest,
    "square": SquareTest,
    "rsqrt": RsqrtTest,
    "tan": TanTest,
    "exp": ExpTest,
    "exp2": Exp2Test,
    "hardswish": HardswishTest,
    "isnan": IsNanTest,
    "isinf": IsInfTest,
    "isfinite": IsFiniteTest,
    "sinc": SincTest,
}


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    # 先获取基础参数解析器
    from libinfiniop.utils import get_args as get_base_args
    import sys
    
    # 创建新的参数解析器，添加 --ops 参数
    parser = argparse.ArgumentParser(description="Test all unary operators", parents=[])
    parser.add_argument(
        "--ops",
        nargs="+",
        choices=list(UNARY_OP_TESTS.keys()),
        default=list(UNARY_OP_TESTS.keys()),
        help="Specify which operators to test (default: all)",
    )
    
    # 解析参数
    args, unknown = parser.parse_known_args()
    
    # 将未知参数传递给基础参数解析器
    if unknown:
        sys.argv = [sys.argv[0]] + unknown
        base_args = get_base_args()
    else:
        # 如果没有其他参数，使用默认值
        sys.argv = [sys.argv[0]]
        base_args = get_base_args()
    
    # 合并参数
    for attr in dir(base_args):
        if not attr.startswith("_") and not hasattr(args, attr):
            setattr(args, attr, getattr(base_args, attr))
    
    # 运行选定的算子测试
    print(f"\n{'='*60}")
    print(f"Testing {len(args.ops)} unary operator(s): {', '.join(args.ops)}")
    print(f"{'='*60}\n")
    
    failed_ops = []
    passed_ops = []
    
    for op_name in args.ops:
        test_class = UNARY_OP_TESTS[op_name]
        print(f"\n{'='*60}")
        print(f"Testing {test_class.OP_NAME} operator")
        print(f"{'='*60}")
        
        try:
            # 创建临时参数对象，传递给测试类
            test_class.DEBUG = args.debug
            test_class.PROFILE = args.profile
            test_class.NUM_PRERUN = args.num_prerun
            test_class.NUM_ITERATIONS = args.num_iterations
            
            # 运行测试
            for device in get_test_devices(args):
                test_operator(device, test_class.test, test_class.TEST_CASES, test_class.TENSOR_DTYPES)
            
            print(f"\033[92m{test_class.OP_NAME} test passed!\033[0m")
            passed_ops.append(op_name)
        except Exception as e:
            print(f"\033[91m{test_class.OP_NAME} test failed: {e}\033[0m")
            failed_ops.append(op_name)
            if args.debug:
                import traceback
                traceback.print_exc()
    
    # 打印总结
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"Total operators: {len(args.ops)}")
    print(f"\033[92mPassed: {len(passed_ops)} - {', '.join(passed_ops)}\033[0m")
    if failed_ops:
        print(f"\033[91mFailed: {len(failed_ops)} - {', '.join(failed_ops)}\033[0m")
    print(f"{'='*60}\n")
    
    if failed_ops:
        exit(1)


if __name__ == "__main__":
    from libinfiniop.utils import get_test_devices, test_operator
    main()
