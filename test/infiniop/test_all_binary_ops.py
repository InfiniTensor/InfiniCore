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


# ==============================================================================
# 算子注册表
# ==============================================================================

# 所有 binary 算子的测试类映射
BINARY_OP_TESTS = {
    "div": DivTest,
    "pow": PowTest,
    "mod": ModTest,
    "max": MaxTest,
    "min": MinTest,
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
