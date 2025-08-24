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
# exp是单输入算子，测试用例包含：形状、输入x的步长、输出c的步长
_TEST_CASES_ = [
    # shape, x_stride, c_stride
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None),  # 测试输入广播/重复场景
    ((13, 4, 4), None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), None),
    ((16, 5632), None, None),
    ((16, 5632), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()  # 输出使用新内存
    INPLACE_X = auto()     # 输出复用输入x的内存（原地操作）


# 为每个测试用例附加inplace选项
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_X,
]

# 组合测试用例：形状+步长+inplace模式
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# 测试支持的数据类型
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# 不同数据类型的误差容限
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-7, "rtol": 1e-7},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def exp(c, x):
    """用PyTorch的exp作为参考实现"""
    if c.shape != x.shape:
        c.resize_(0)
    torch.exp(x, out=c)


def test(
    handle,
    device,
    shape,
    x_stride=None,
    c_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    # 创建输入张量x
    x = TestTensor(shape, x_stride, dtype, device)
    # 根据inplace模式创建输出张量c
    if inplace == Inplace.INPLACE_X:
        # 原地操作：c复用x的内存（需步长匹配）
        if x_stride != c_stride:
            return
        c = x
    else:
        # 非原地操作：c使用新内存
        c = TestTensor(shape, c_stride, dtype, device, mode="ones")

    # 跳过广播场景（如需支持广播可移除）
    if c.is_broadcast():
        return

    # 打印测试信息
    print(
        f"Testing Exp on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} c_stride:{c_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )
    # 用PyTorch计算参考结果
    exp(c.torch_tensor(), x.torch_tensor())

    if sync is not None:
        sync()

    # 创建exp算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateExpDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,  # 输出张量描述符
            x.descriptor   # 输入张量描述符（单输入）
        )
    )

    # 销毁张量描述符缓存（模拟实际场景）
    for tensor in [x, c]:
        tensor.destroy_desc()

    # 分配工作空间
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetExpWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    # 定义自定义库的exp调用函数
    def lib_exp():
        check_error(
            LIBINFINIOP.infiniopExp(
                descriptor,
                workspace.data(),
                workspace.size(),
                c.data(),  # 输出数据地址
                x.data(),  # 输入数据地址（单输入）
                None       # 额外参数
            )
        )

    # 执行自定义库的exp算子
    lib_exp()
    # 验证结果正确性
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)


    # 性能 profiling（对比自定义库与PyTorch性能）
    if PROFILE:
        profile_operation("PyTorch", lambda: exp(c.torch_tensor(), x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_exp(), device, NUM_PRERUN, NUM_ITERATIONS)
    
    # 销毁算子描述符
    check_error(LIBINFINIOP.infiniopDestroyExpDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    # # 解析命令行参数配置测试
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    # 在所有测试设备上执行测试
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
    