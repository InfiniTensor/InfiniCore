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
    # shape, input_stride, output_stride
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((16, 5632), None, None),
    ((4, 4, 5632), None, None),
]

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()

_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]

# HardTanh 特有的参数测试组合 (min_val, max_val)
_PARAM_CASES = [
    (-1.0, 1.0),
    (0.0, 6.0), # 类似于 ReLU6
    (-2.5, 2.5),
]

# 组合所有测试用例：shape + inplace + params
_TEST_CASES = [
    test_case + (inplace_item, p_min, p_max)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
    for p_min, p_max in _PARAM_CASES
]

_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def test(
    handle,
    device,
    shape,
    input_stride=None,
    output_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    min_val=-1.0,
    max_val=1.0,
    dtype=torch.float16,
    sync=None,
):
    input = TestTensor(shape, input_stride, dtype, device)
    if inplace == Inplace.INPLACE:
        if input_stride != output_stride:
            return
        output = input
    else:
        output = TestTensor(shape, output_stride, dtype, device, mode="ones")

    if output.is_broadcast():
        return

    print(
        f"Testing HardTanh on {InfiniDeviceNames[device]} | shape:{shape} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace} range:[{min_val}, {max_val}]"
    )

    # 计算 PyTorch 真值
    new_output = torch.nn.functional.hardtanh(input.torch_tensor(), min_val=min_val, max_val=max_val)
    output.update_torch_tensor(new_output)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()

    check_error(
        LIBINFINIOP.infiniopCreateHardTanhDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            input.descriptor,
            c_float(min_val),
            c_float(max_val),
        )
    )

    for tensor in [input, output]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetHardTanhWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output.device)

    def lib_hardtanh():
        check_error(
            LIBINFINIOP.infiniopHardTanh(
                descriptor,
                workspace.data(),
                workspace.size(),
                output.data(),
                input.data(),
                None,
            )
        )

    lib_hardtanh()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)
    
    assert torch.allclose(
        output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol
    )

    if PROFILE:
        profile_operation("PyTorch", lambda: torch.nn.functional.hardtanh(input.torch_tensor(), min_val, max_val), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("   lib", lambda: lib_hardtanh(), device, NUM_PRERUN, NUM_ITERATIONS)
        
    check_error(LIBINFINIOP.infiniopDestroyHardTanhDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mHardTanh Test passed!\033[0m")