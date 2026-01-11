import torch
import ctypes
from ctypes import c_uint64, c_float, c_double
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

# ==============================================================================
# Configuration
# ==============================================================================
# 格式: (M, N, D, x1_stride, x2_stride)
# x1: (M, D), x2: (N, D), out: (M, N)
_TEST_CASES_DATA = [
    (5, 6, 3, None, None),
    (32, 64, 128, None, None),
    (32, 64, 128, (256, 1), (256, 1)), # 测试带步长的输入
    (10, 7, 5, None, None),
]

_TENSOR_DTYPES = [InfiniDtype.F32] # cdist 通常对精度敏感，初测建议用 F32

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-4},
}

_P_VALUES = [1.0, 2.0, float("inf")] # 不同的 p 范数测试

_TEST_CASES = [
    test_case + (p_val,)
    for test_case in _TEST_CASES_DATA
    for p_val in _P_VALUES
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100

def test(
    handle,
    device,
    M, N, D,
    x1_stride=None,
    x2_stride=None,
    p=2.0,
    dtype=InfiniDtype.F32,
    sync=None,
):
    print(
        f"Testing Cdist on {InfiniDeviceNames[device]} with M:{M}, N:{N}, D:{D}, p:{p}, dtype:{InfiniDtypeNames[dtype]}"
    )

    # 1. 准备输入输出形状
    x1_shape = (M, D)
    x2_shape = (N, D)
    out_shape = (M, N)

    # 2. 准备输入 Tensor
    x1 = TestTensor(x1_shape, x1_stride, dtype, device)
    x2 = TestTensor(x2_shape, x2_stride, dtype, device)
    out = TestTensor(out_shape, None, dtype, device)

    # 3. 使用 PyTorch 计算参考答案
    # torch.cdist 要求输入至少是 2D
    ans = torch.cdist(x1.torch_tensor(), x2.torch_tensor(), p=p)

    if sync is not None:
        sync()

    # 4. 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    # 注意：这里假设 C 接口名为 infiniopCreateCdistDescriptor
    check_error(
        LIBINFINIOP.infiniopCreateCdistDescriptor(
            handle, 
            ctypes.byref(descriptor), 
            out.descriptor, 
            x1.descriptor, 
            x2.descriptor, 
            c_double(p) # 通常 p 使用 double 或 float 传递
        )
    )

    # 销毁临时描述符以防内核错误引用（沿用 addcmul 风格）
    for t in [x1, x2, out]:
        t.destroy_desc()

    # 5. Workspace 准备
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetCdistWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x1.device)

    # 6. 执行函数定义
    def lib_cdist():
        check_error(
            LIBINFINIOP.infiniopCdist(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                x1.data(),
                x2.data(),
                None, # stream
            )
        )

    # 7. 运行
    lib_cdist()

    if sync is not None:
        sync()

    # 8. 验证结果
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    # 9. 性能分析
    if PROFILE:
        profile_operation("PyTorch", lambda: torch.cdist(x1.torch_tensor(), x2.torch_tensor(), p=p), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("   lib", lambda: lib_cdist(), device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyCdistDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mCdist tests passed!\033[0m")