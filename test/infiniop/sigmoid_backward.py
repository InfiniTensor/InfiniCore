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
#  Configuration
# ==============================================================================
# sigmoid_backward 是双输入算子：grad_output 和 input
# shape,grad_stride=None,x_stride=None,c_stride=None,
_TEST_CASES_ = [
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (16, 4, 1), (16, 4, 1), (16, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (4, 0, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()

_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_X,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

_TENSOR_DTYPES = [
    InfiniDtype.F16,
    InfiniDtype.F32,
    InfiniDtype.BF16
]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def sigmoid_backward_ref(input, grad_output):
    input.requires_grad_(True)
    output = torch.nn.functional.sigmoid(input)
    output.backward(grad_output)
    return input.grad
    


def test(
    handle,
    device,
    shape,
    grad_stride=None,
    x_stride=None,
    c_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    # 输入
    grad_output = TestTensor(shape, grad_stride, dtype, device)
    input_x = TestTensor(shape, x_stride, dtype, device)

    if inplace == Inplace.INPLACE_X:
        if c_stride != grad_stride:
            return
        grad_input = grad_output
    else:
        grad_input = TestTensor(shape, c_stride, dtype, device, mode="ones")

    if grad_input.is_broadcast():
        return

    print(
        f"Testing sigmoid_backward on {InfiniDeviceNames[device]} "
        f"shape:{shape} grad_stride:{grad_stride} x_stride:{x_stride} c_stride:{c_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    # PyTorch参考输出
    grad_input_torch = sigmoid_backward_ref(input_x.torch_tensor(), grad_output.torch_tensor())
    grad_input.torch_tensor().copy_(grad_input_torch)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSigmoidBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
            grad_input.descriptor,  # 输出 grad_input
            grad_output.descriptor, # 输入 grad_output
            input_x.descriptor      # 输入 x
        )
    )

    for tensor in [grad_output, input_x, grad_input]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSigmoidBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, grad_input.device)

    def lib_sigmoid_backward():
        check_error(
            LIBINFINIOP.infiniopSigmoidBackward(
                descriptor,
                workspace.data(),
                workspace.size(),
                grad_input.data(),
                grad_output.data(),
                input_x.data(),
                None
            )
        )

    lib_sigmoid_backward()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(grad_input.actual_tensor(), grad_input.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation("PyTorch", lambda: sigmoid_backward_ref(input_x.torch_tensor(), grad_output.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_sigmoid_backward(), device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroySigmoidBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
