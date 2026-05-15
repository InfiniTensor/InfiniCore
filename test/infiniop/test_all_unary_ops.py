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
    "neg": NegTest,
    "reciprocal": ReciprocalTest,
    "round": RoundTest,
    "sign": SignTest,
    "sinh": SinhTest,
    "sqrt": SqrtTest,
    "tan": TanTest,
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
