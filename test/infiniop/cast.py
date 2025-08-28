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
# cast是单输入算子，测试用例包含：形状、输入x的步长、输出c的步长
_TEST_CASES_ = [
    # shape, x_stride, c_stride
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None),
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


FLOAT_DTYPES = [
    InfiniDtype.F16,
    InfiniDtype.F32,
    InfiniDtype.F64,
]

INTER_DTYPES = [
    InfiniDtype.I32,
    InfiniDtype.I64,
    InfiniDtype.U32,
    InfiniDtype.U64,
]

# 测试支持的数据类型, 全部类型转换为浮点类型：
# 浮点类型转换为浮点类型，整数类型转换为浮点类型
_TENSOR_DTYPES = [
    (ftype, ttype) for ftype in FLOAT_DTYPES + INTER_DTYPES
    for ttype in FLOAT_DTYPES
]

# 整数类型之间相互转换
_TENSOR_DTYPES.extend([
    (ftype, ttype) for ftype in INTER_DTYPES
    for ttype in INTER_DTYPES
])

# _TENSOR_DTYPES = [
#     #   输入类型       输出类型
#     (InfiniDtype.F16, InfiniDtype.F16),
#     (InfiniDtype.F32, InfiniDtype.F16),
#     (InfiniDtype.F64, InfiniDtype.F16),
#     (InfiniDtype.I32, InfiniDtype.F16),
#     (InfiniDtype.I64, InfiniDtype.F16),
#     (InfiniDtype.U32, InfiniDtype.F16),
#     (InfiniDtype.U64, InfiniDtype.F16),

#     # (InfiniDtype.F16, InfiniDtype.F32),
#     # (InfiniDtype.F64, InfiniDtype.F32),
#     # (InfiniDtype.I32, InfiniDtype.F32),
#     # (InfiniDtype.I64, InfiniDtype.F32),
#     # ……
# ]

# 不同数据类型的误差容限
_TOLERANCE_MAP = {
    (ftype, ttype): {"atol": 1e-3, "rtol": 1e-3}
    for ftype in FLOAT_DTYPES + INTER_DTYPES
    for ttype in FLOAT_DTYPES
}

# 添加整数类型之间的转换及误差
_TOLERANCE_MAP.update({
    (ftype, ttype): {"atol": 0, "rtol": 0}
    for ftype in INTER_DTYPES
    for ttype in INTER_DTYPES
})

# 特别处理 F64 浮点类型转换为 F16 浮点类型
_TOLERANCE_MAP.update({
    (InfiniDtype.F64, InfiniDtype.F16): {"atol": 1e-3, "rtol": 1e-3}
})


DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# 方案一：bug在于 GPU 上不支持 UInt32/UInt64 的 copy_
# def cast(c: torch.Tensor, x: torch.Tensor):
#     """
#     PyTorch参考实现Cast
#     c: 输出张量
#     x: 输入张量
#     dtype: torch数据类型（如 torch.float32, torch.int32）
#     """
#     # 在 CPU 上进行参考计算，确保类型和行为一致
#     # 打印x的设备信息 ['cuda:0', 'cpu']
#     if not x.device.type.startswith('cpu') and c.dtype in [torch.uint32, torch.uint64]:
#         x = x.cpu().to(c.dtype)
#     c.copy_(x.to(c.device))

# 方案二：
# 避开了 GPU UInt32/UInt64 的限制
# 直接在 CPU 上用 NumPy 做类型转换，兼容非连续张量
def cast(c: torch.Tensor, x: torch.Tensor):
    if not x.device.type.startswith('cpu') and c.dtype in [torch.uint32, torch.uint64]:
        x_np = x.cpu().numpy()
        if c.dtype == torch.uint32:
            c_np = x_np.astype('uint32')
        elif c.dtype == torch.uint64:
            c_np = x_np.astype('uint64')
        c.copy_(torch.from_numpy(c_np))
    else:
        c.copy_(x.to(c.dtype))


def test(
    handle,
    device,
    shape,
    x_stride=None,
    c_stride=None,
    dtype=(InfiniDtype.F32, InfiniDtype.F64),
    sync=None,
):
    # 创建输入张量x
    
    x = TestTensor(shape, x_stride, dtype[0], device)
    c = TestTensor(shape, c_stride, dtype[1], device)

    # 打印测试信息
    print(
        f"Testing Cast on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} c_stride:{c_stride} "
        f"Cast dtype from {InfiniDtypeNames[dtype[0]]} to {InfiniDtypeNames[dtype[1]]}"
    )
    # 用PyTorch计算参考结果
    cast(c.torch_tensor(), x.torch_tensor())
    if sync is not None:
        sync()

    # 创建cast算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateCastDescriptor(
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
        LIBINFINIOP.infiniopGetCastWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    # 定义自定义库的cast调用函数
    def lib_cast():
        check_error(
            LIBINFINIOP.infiniopCast(
                descriptor,
                workspace.data(),
                workspace.size(),
                c.data(),  # 输出数据地址
                x.data(),  # 输入数据地址（单输入）
                None       # 额外参数
            )
        )

    # 执行自定义库的cast算子
    lib_cast()

    # 验证结果正确性
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    
    # assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    
    actual = c.actual_tensor()
    expected = c.torch_tensor()
    if expected.dtype in [torch.float16, torch.float32, torch.float64]:
        assert torch.allclose(actual, expected, atol=atol, rtol=rtol)
    else:
        assert torch.equal(actual, expected), \
            f"Integer cast mismatch!\nExpected:\n{expected}\nActual:\n{actual}"

    # 性能 profiling（对比自定义库与PyTorch性能）
    if PROFILE:
        profile_operation("PyTorch", lambda: cast(c.torch_tensor(), x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_cast(), device, NUM_PRERUN, NUM_ITERATIONS)
    
    # 销毁算子描述符
    check_error(LIBINFINIOP.infiniopDestroyCastDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    # # 解析命令行参数配置测试
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    # 在所有测试设备上执行测试
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")