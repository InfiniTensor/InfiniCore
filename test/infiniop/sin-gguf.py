import torch
import ctypes
from ctypes import c_uint64
from gguf import GGUFReader
from enum import Enum, auto

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

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()

# 支持的数据类型
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

# PyTorch参考实现
def sin(output, input):
    if output.shape != input.shape:
        output.resize_(input.shape)
    torch.sin(input, out=output)

# 从 gguf 文件加载测试用例
def load_test_cases_from_gguf(filepath):
    reader = GGUFReader(filepath)
    tensors = reader.tensors

    test_cases = []
    for tensor in tensors:
        data = tensor.data
        shape = data.shape
        torch_tensor = torch.from_numpy(data.copy())
        x_stride = torch_tensor.stride()
        c_stride = None

        for inplace in [Inplace.OUT_OF_PLACE, Inplace.INPLACE_X]:
            test_cases.append((shape, x_stride, c_stride, inplace, torch_tensor))

    return test_cases

def test(
    handle,
    device,
    shape,
    x_stride=None,
    c_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    torch_tensor=None,
    dtype=torch.float16,
    sync=None,
):
    x = TestTensor(shape, x_stride, dtype, device, mode="manual", set_tensor=torch_tensor)
    if inplace == Inplace.INPLACE_X:
        # if x_stride != c_stride:
        #     return
        c = x
    else:
        c = TestTensor(shape, c_stride, dtype, device, mode="ones")

    if c.is_broadcast():
        return

    print(
        f"Testing sin on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} "
        f"c_stride:{c_stride} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    sin(c.torch_tensor(), x.torch_tensor())
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSinDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            x.descriptor
        )
    )

    for tensor in [x, c]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetSinWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
    workspace = TestWorkspace(workspace_size.value, c.device)

    def lib_sin():
        check_error(
            LIBINFINIOP.infiniopSin(
                descriptor,
                workspace.data(),
                workspace.size(),
                c.data(),
                x.data(),
                None
            )
        )

    lib_sin()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    actual = c.actual_tensor()
    expected = c.torch_tensor()
    if DEBUG:
        debug(actual, expected, atol=atol, rtol=rtol)

    assert torch.allclose(actual, expected, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation("PyTorch", lambda: sin(c.torch_tensor(), x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_sin, device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroySinDescriptor(descriptor))

if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # gguf 文件路径示例，按实际情况修改
    _TEST_CASES = {
        InfiniDtype.F16: load_test_cases_from_gguf("T1-1-1/sin/sin_bf16.gguf"),
        InfiniDtype.F32: load_test_cases_from_gguf("T1-1-1/sin/sin_f32.gguf"),
        InfiniDtype.BF16: load_test_cases_from_gguf("T1-1-1/sin/sin_bf16.gguf"),
    }


    for device in get_test_devices(args):
        for dtype in _TEST_CASES:
            test_operator(device, test, _TEST_CASES[dtype], [dtype])

    print("\033[92mTest passed!\033[0m")
