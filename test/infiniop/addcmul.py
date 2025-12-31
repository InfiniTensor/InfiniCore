import torch
import ctypes
from ctypes import c_uint64, c_float
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
# Configuration
# ==============================================================================
_TEST_CASES_ = [
    # shape, input_stride, t1_stride, t2_stride
    ((3, 3), None, None, None),
    ((32, 512), None, None, None),
    ((32, 512), (1024, 1), (1024, 1), (1024, 1)),
    ((16, 32, 64), None, None, None),
    ((8, 1, 1024), None, None, None), # 包含广播形状的潜在测试
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_INPUT = auto()

_INPLACE = [Inplace.OUT_OF_PLACE, Inplace.INPLACE_INPUT]
_VALUES = [1.0, 0.5, -2.0] # 测试不同的 value 系数

_TEST_CASES = [
    test_case + (inplace_item, value)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
    for value in _VALUES
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100

def test(
    handle,
    device,
    shape,
    input_stride=None,
    t1_stride=None,
    t2_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    value=1.0,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing Addcmul on {InfiniDeviceNames[device]} with shape:{shape} value:{value} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    # 准备输入 Tensor
    input_tensor = TestTensor(shape, input_stride, dtype, device)
    t1 = TestTensor(shape, t1_stride, dtype, device)
    t2 = TestTensor(shape, t2_stride, dtype, device)

    # 使用 PyTorch 计算参考答案
    # out = input + value * t1 * t2
    ans = torch.addcmul(input_tensor.torch_tensor(), t1.torch_tensor(), t2.torch_tensor(), value=value)

    if inplace == Inplace.INPLACE_INPUT:
        out = input_tensor
    else:
        out = TestTensor(shape, None, dtype, device)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    # 注意：根据之前的定义，Create 接口接收 value
    check_error(
        LIBINFINIOP.infiniopCreateAddcmulDescriptor(
            handle, 
            ctypes.byref(descriptor), 
            out.descriptor, 
            input_tensor.descriptor, 
            t1.descriptor, 
            t2.descriptor, 
            c_float(value)
        )
    )

    # 销毁临时描述符以防内核错误引用
    for t in [input_tensor, t1, t2, out]:
        t.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAddcmulWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, input_tensor.device)

    def lib_addcmul():
        check_error(
            LIBINFINIOP.infiniopAddcmul(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                input_tensor.data(),
                t1.data(),
                t2.data(),
                None,
            )
        )

    lib_addcmul()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation("PyTorch", lambda: torch.addcmul(input_tensor.torch_tensor(), t1.torch_tensor(), t2.torch_tensor(), value=value), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_addcmul(), device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyAddcmulDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mAddcmul tests passed!\033[0m")