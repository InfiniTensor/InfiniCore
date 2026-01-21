"""
统一测试所有 Binary 算子

这个文件包含所有 binary 算子的测试，方便统一管理和运行。
可以通过命令行参数选择运行哪些算子，或者运行所有算子。

使用方法:
    # 运行所有 binary 算子测试
    python test_all_binary_ops.py
    
    # 只运行 div 和 pow 算子
    python test_all_binary_ops.py --ops div pow
    
    # 运行特定设备上的测试
    python test_all_binary_ops.py --cpu --nvidia
"""

import torch
import argparse
from libinfiniop import InfiniDtype, TestTensor
from libinfiniop.binary_test_base import BinaryTestBase


# ==============================================================================
# 所有 Binary 算子的测试类定义
# ==============================================================================

class DivTest(BinaryTestBase):
    OP_NAME = "Div"
    OP_NAME_LOWER = "div"
    
    @staticmethod
    def torch_op(c, a, b):
        torch.div(a, b, out=c)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        # For division, ensure b doesn't contain zeros
        return TestTensor(shape, b_stride, dtype, device, scale=2, bias=0.1)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class FloorDivideTest(BinaryTestBase):
    OP_NAME = "FloorDivide"
    OP_NAME_LOWER = "floor_divide"
    
    @staticmethod
    def torch_op(c, a, b):
        torch.floor_divide(a, b, out=c)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        # For division, ensure b doesn't contain zeros
        return TestTensor(shape, b_stride, dtype, device, scale=2, bias=0.1)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class PowTest(BinaryTestBase):
    OP_NAME = "Pow"
    OP_NAME_LOWER = "pow"
    
    @staticmethod
    def torch_op(c, a, b):
        torch.pow(a, b, out=c)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        # Avoid negative bases and very large exponents
        return TestTensor(shape, a_stride, dtype, device, mode="random", scale=5.0, bias=0.1)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device, mode="random", scale=3.0, bias=0.1)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-3},
    }
    
    EQUAL_NAN = True


class CopySignTest(BinaryTestBase):
    OP_NAME = "CopySign"
    OP_NAME_LOWER = "copysign"
    
    @staticmethod
    def torch_op(c, a, b):
        torch.copysign(a, b, out=c)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        # Generate values with various magnitudes
        return TestTensor(shape, a_stride, dtype, device, mode="random", scale=10.0, bias=-5.0)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        # Generate values with various signs
        return TestTensor(shape, b_stride, dtype, device, mode="random", scale=10.0, bias=-5.0)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    EQUAL_NAN = True


class HypotTest(BinaryTestBase):
    OP_NAME = "Hypot"
    OP_NAME_LOWER = "hypot"
    
    @staticmethod
    def torch_op(c, a, b):
        torch.hypot(a, b, out=c)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device, mode="random", scale=10.0, bias=-5.0)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device, mode="random", scale=10.0, bias=-5.0)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class Atan2Test(BinaryTestBase):
    OP_NAME = "Atan2"
    OP_NAME_LOWER = "atan2"
    
    @staticmethod
    def torch_op(c, a, b):
        torch.atan2(a, b, out=c)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        # For atan2, avoid zeros in denominator (b)
        return TestTensor(shape, b_stride, dtype, device, scale=2, bias=0.1)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class ModTest(BinaryTestBase):
    OP_NAME = "Mod"
    OP_NAME_LOWER = "mod"
    
    @staticmethod
    def torch_op(c, a, b):
        torch.remainder(a, b, out=c)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        # Avoid zeros
        return TestTensor(shape, b_stride, dtype, device, scale=2, bias=0.1)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class RemainderTest(BinaryTestBase):
    OP_NAME = "Remainder"
    OP_NAME_LOWER = "remainder"
    
    @staticmethod
    def torch_op(c, a, b):
        torch.remainder(a, b, out=c)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        # Avoid zeros
        return TestTensor(shape, b_stride, dtype, device, scale=2, bias=0.1)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class MaxTest(BinaryTestBase):
    OP_NAME = "Max"
    OP_NAME_LOWER = "max"
    
    @staticmethod
    def torch_op(c, a, b):
        torch.maximum(a, b, out=c)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class MinTest(BinaryTestBase):
    OP_NAME = "Min"
    OP_NAME_LOWER = "min"
    
    @staticmethod
    def torch_op(c, a, b):
        torch.minimum(a, b, out=c)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class FmaxTest(BinaryTestBase):
    OP_NAME = "Fmax"
    OP_NAME_LOWER = "fmax"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.fmax ignores NaN: if one is NaN, return the other
        result = torch.fmax(a, b)
        c.copy_(result)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class FminTest(BinaryTestBase):
    OP_NAME = "Fmin"
    OP_NAME_LOWER = "fmin"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.fmin ignores NaN: if one is NaN, return the other
        result = torch.fmin(a, b)
        c.copy_(result)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class GtTest(BinaryTestBase):
    OP_NAME = "Gt"
    OP_NAME_LOWER = "gt"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.gt returns bool, convert to float (1.0 or 0.0) to match our implementation
        result = torch.gt(a, b)
        c.copy_(result.float())
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class LtTest(BinaryTestBase):
    OP_NAME = "Lt"
    OP_NAME_LOWER = "lt"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.lt returns bool, convert to float (1.0 or 0.0) to match our implementation
        result = torch.lt(a, b)
        c.copy_(result.float())
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class GeTest(BinaryTestBase):
    OP_NAME = "Ge"
    OP_NAME_LOWER = "ge"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.ge returns bool, convert to float (1.0 or 0.0) to match our implementation
        result = torch.ge(a, b)
        c.copy_(result.float())
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class LeTest(BinaryTestBase):
    OP_NAME = "Le"
    OP_NAME_LOWER = "le"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.le returns bool, convert to float (1.0 or 0.0) to match our implementation
        result = torch.le(a, b)
        c.copy_(result.float())
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class EqTest(BinaryTestBase):
    OP_NAME = "Eq"
    OP_NAME_LOWER = "eq"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.eq returns bool, convert to float (1.0 or 0.0) to match our implementation
        result = torch.eq(a, b)
        c.copy_(result.float())
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class NeTest(BinaryTestBase):
    OP_NAME = "Ne"
    OP_NAME_LOWER = "ne"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.ne returns bool, convert to float (1.0 or 0.0) to match our implementation
        result = torch.ne(a, b)
        c.copy_(result.float())
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class LogicalAndTest(BinaryTestBase):
    OP_NAME = "LogicalAnd"
    OP_NAME_LOWER = "logical_and"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.logical_and returns bool, convert to float (1.0 or 0.0) to match our implementation
        result = torch.logical_and(a, b)
        c.copy_(result.float())
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class LogicalOrTest(BinaryTestBase):
    OP_NAME = "LogicalOr"
    OP_NAME_LOWER = "logical_or"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.logical_or returns bool, convert to float (1.0 or 0.0) to match our implementation
        result = torch.logical_or(a, b)
        c.copy_(result.float())
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class LogicalXorTest(BinaryTestBase):
    OP_NAME = "LogicalXor"
    OP_NAME_LOWER = "logical_xor"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.logical_xor returns bool, convert to float (1.0 or 0.0) to match our implementation
        result = torch.logical_xor(a, b)
        c.copy_(result.float())
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    EQUAL_NAN = True


class BitwiseAndTest(BinaryTestBase):
    OP_NAME = "BitwiseAnd"
    OP_NAME_LOWER = "bitwise_and"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.bitwise_and only supports integral types
        result = torch.bitwise_and(a, b)
        c.copy_(result)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        # Use default TestTensor (utils.py now handles correct ranges for integral types)
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        # Use default TestTensor (utils.py now handles correct ranges for integral types)
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.I32: {"atol": 0, "rtol": 0},
        InfiniDtype.I64: {"atol": 0, "rtol": 0},
        InfiniDtype.U8: {"atol": 0, "rtol": 0},
    }
    
    # Bitwise operations only support integral types
    TENSOR_DTYPES = [InfiniDtype.I32, InfiniDtype.I64, InfiniDtype.U8]
    
    EQUAL_NAN = True


class BitwiseOrTest(BinaryTestBase):
    OP_NAME = "BitwiseOr"
    OP_NAME_LOWER = "bitwise_or"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.bitwise_or only supports integral types
        result = torch.bitwise_or(a, b)
        c.copy_(result)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        # Use default TestTensor (utils.py now handles correct ranges for integral types)
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        # Use default TestTensor (utils.py now handles correct ranges for integral types)
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.I32: {"atol": 0, "rtol": 0},
        InfiniDtype.I64: {"atol": 0, "rtol": 0},
        InfiniDtype.U8: {"atol": 0, "rtol": 0},
    }
    
    # Bitwise operations only support integral types
    TENSOR_DTYPES = [InfiniDtype.I32, InfiniDtype.I64, InfiniDtype.U8]
    
    EQUAL_NAN = True


class BitwiseXorTest(BinaryTestBase):
    OP_NAME = "BitwiseXor"
    OP_NAME_LOWER = "bitwise_xor"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.bitwise_xor only supports integral types
        result = torch.bitwise_xor(a, b)
        c.copy_(result)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        # Use default TestTensor (utils.py now handles correct ranges for integral types)
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        # Use default TestTensor (utils.py now handles correct ranges for integral types)
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.I32: {"atol": 0, "rtol": 0},
        InfiniDtype.I64: {"atol": 0, "rtol": 0},
        InfiniDtype.U8: {"atol": 0, "rtol": 0},
    }
    
    # Bitwise operations only support integral types
    TENSOR_DTYPES = [InfiniDtype.I32, InfiniDtype.I64, InfiniDtype.U8]
    
    EQUAL_NAN = True


class BitwiseLeftShiftTest(BinaryTestBase):
    OP_NAME = "BitwiseLeftShift"
    OP_NAME_LOWER = "bitwise_left_shift"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.bitwise_left_shift only supports integral types
        result = torch.bitwise_left_shift(a, b)
        c.copy_(result)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        # Use default TestTensor (utils.py now handles correct ranges for integral types)
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        # For shift operations, b should be non-negative and within reasonable range
        # Generate shift amounts between 0 and bit_width-1 for each type
        if dtype == InfiniDtype.U8:
            return TestTensor(shape, b_stride, dtype, device, randint_low=0, randint_high=8)
        elif dtype == InfiniDtype.I32:
            return TestTensor(shape, b_stride, dtype, device, randint_low=0, randint_high=32)
        elif dtype == InfiniDtype.I64:
            return TestTensor(shape, b_stride, dtype, device, randint_low=0, randint_high=64)
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.I32: {"atol": 0, "rtol": 0},
        InfiniDtype.I64: {"atol": 0, "rtol": 0},
        InfiniDtype.U8: {"atol": 0, "rtol": 0},
    }
    
    # Bitwise operations only support integral types
    TENSOR_DTYPES = [InfiniDtype.I32, InfiniDtype.I64, InfiniDtype.U8]
    
    EQUAL_NAN = True


class BitwiseRightShiftTest(BinaryTestBase):
    OP_NAME = "BitwiseRightShift"
    OP_NAME_LOWER = "bitwise_right_shift"
    
    @staticmethod
    def torch_op(c, a, b):
        # torch.bitwise_right_shift only supports integral types
        result = torch.bitwise_right_shift(a, b)
        c.copy_(result)
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        # Use default TestTensor (utils.py now handles correct ranges for integral types)
        return TestTensor(shape, a_stride, dtype, device)
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        # For shift operations, b should be non-negative and within reasonable range
        # Generate shift amounts between 0 and bit_width-1 for each type
        if dtype == InfiniDtype.U8:
            return TestTensor(shape, b_stride, dtype, device, randint_low=0, randint_high=8)
        elif dtype == InfiniDtype.I32:
            return TestTensor(shape, b_stride, dtype, device, randint_low=0, randint_high=32)
        elif dtype == InfiniDtype.I64:
            return TestTensor(shape, b_stride, dtype, device, randint_low=0, randint_high=64)
        return TestTensor(shape, b_stride, dtype, device)
    
    TOLERANCE_MAP = {
        InfiniDtype.I32: {"atol": 0, "rtol": 0},
        InfiniDtype.I64: {"atol": 0, "rtol": 0},
        InfiniDtype.U8: {"atol": 0, "rtol": 0},
    }
    
    # Bitwise operations only support integral types
    TENSOR_DTYPES = [InfiniDtype.I32, InfiniDtype.I64, InfiniDtype.U8]
    
    EQUAL_NAN = True


# ==============================================================================
# 算子注册表
# ==============================================================================

# 所有 binary 算子的测试类映射
BINARY_OP_TESTS = {
    "div": DivTest,
    "floor_divide": FloorDivideTest,
    "pow": PowTest,
    "copysign": CopySignTest,
    "hypot": HypotTest,
    "atan2": Atan2Test,
    "mod": ModTest,
    "remainder": RemainderTest,
    "max": MaxTest,
    "min": MinTest,
    "fmax": FmaxTest,
    "fmin": FminTest,
    "gt": GtTest,
    "lt": LtTest,
    "ge": GeTest,
    "le": LeTest,
    "eq": EqTest,
    "ne": NeTest,
    "logical_and": LogicalAndTest,
    "logical_or": LogicalOrTest,
    "logical_xor": LogicalXorTest,
    "bitwise_and": BitwiseAndTest,
    "bitwise_or": BitwiseOrTest,
    "bitwise_xor": BitwiseXorTest,
    "bitwise_left_shift": BitwiseLeftShiftTest,
    "bitwise_right_shift": BitwiseRightShiftTest,
}


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    # 先获取基础参数解析器
    from libinfiniop.utils import get_args as get_base_args
    import sys
    
    # 创建新的参数解析器，添加 --ops 参数
    parser = argparse.ArgumentParser(description="Test all binary operators", parents=[])
    parser.add_argument(
        "--ops",
        nargs="+",
        choices=list(BINARY_OP_TESTS.keys()),
        default=list(BINARY_OP_TESTS.keys()),
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
    print(f"Testing {len(args.ops)} binary operator(s): {', '.join(args.ops)}")
    print(f"{'='*60}\n")
    
    failed_ops = []
    passed_ops = []
    
    for op_name in args.ops:
        test_class = BINARY_OP_TESTS[op_name]
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
