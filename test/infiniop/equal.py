import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
_TEST_CASES_ = [
    # shape, a_stride, b_stride, c_stride
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((13, 16, 2), (128, 4, 1), (0, 2, 1), (64, 4, 1)),
    ((13, 16, 2), (128, 4, 1), (2, 0, 1), (64, 4, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]

# Equal 算子通常不支持 Inplace (输入Float vs 输出Bool，内存大小不同)
class Inplace(Enum):
    OUT_OF_PLACE = auto()

_INPLACE = [
    Inplace.OUT_OF_PLACE,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# 测试的输入数据类型
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16, InfiniDtype.I32, InfiniDtype.I64]

# 容差设置 (对于 Bool 比较，通常要求完全匹配)
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.F32: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
    InfiniDtype.I32: {"atol": 0, "rtol": 0},
    InfiniDtype.I64: {"atol": 0, "rtol": 0},
    InfiniDtype.BOOL: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

# PyTorch 标准实现
def equal_func(c, a, b):
    torch.eq(a, b, out=c)

def test(
    handle,
    device,
    shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    # 输入 Tensor 使用指定的 dtype (如 float16)
    a = TestTensor(shape, a_stride, dtype, device)
    b = TestTensor(shape, b_stride, dtype, device)
    
    # [关键修改] 输出 Tensor 强制使用 Bool 类型
    # 注意：这里 c_stride 如果是按字节计算的，对于 Bool 类型通常是 1 byte
    c = TestTensor(shape, c_stride, InfiniDtype.BOOL, device)

    if c.is_broadcast():
        return

    print(
        f"Testing Equal on {InfiniDeviceNames[device]} with shape:{shape} a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} "
        f"input_dtype:{InfiniDtypeNames[dtype]} output_dtype:BOOL"
    )

    # 运行 PyTorch 对照组
    equal_func(c.torch_tensor(), a.torch_tensor(), b.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    
    # [关键修改] 调用 Equal 的 Create 函数
    check_error(
        LIBINFINIOP.infiniopCreateEqualDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor, # Output (Bool)
            a.descriptor, # Input A
            b.descriptor, # Input B
        )
    )

    # Invalidate descriptors
    for tensor in [a, b, c]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetEqualWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    def lib_equal():
        check_error(
            LIBINFINIOP.infiniopEqual(
                descriptor,
                workspace.data(),
                workspace.size(),
                c.data(),
                a.data(),
                b.data(),
                None,
            )
        )

    lib_equal()

    # 使用 Bool 类型的容差 (实际上就是全等)
    atol, rtol = get_tolerance(_TOLERANCE_MAP, InfiniDtype.BOOL)
    
    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    
    # 验证结果
    assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: equal_func(c.torch_tensor(), a.torch_tensor(), b.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_equal(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
        
    check_error(LIBINFINIOP.infiniopDestroyEqualDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
