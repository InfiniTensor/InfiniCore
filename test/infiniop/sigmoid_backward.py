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
# 和 ReLU 保持一致的形状/步幅用例
_TEST_CASES_ = [
    # shape, input_stride, grad_output_stride, grad_input_stride
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_INPUT = auto()
    INPLACE_GRAD_OUTPUT = auto()


# 每个 case 都测试三种 inplace 方式
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_INPUT,
    Inplace.INPLACE_GRAD_OUTPUT,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# 数据类型
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Sigmoid backward 容差（略宽于 ReLU）
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 2e-3, "rtol": 2e-3},
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    InfiniDtype.BF16: {"atol": 3e-2, "rtol": 3e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def sigmoid_backward(grad_input, input_tensor, grad_output):
    """
    PyTorch reference implementation of Sigmoid backward.

    Given:
      y = sigmoid(x) = 1 / (1 + exp(-x))
    Then:
      dL/dx = dL/dy * y * (1 - y)
    """
    s = torch.sigmoid(input_tensor)
    result = grad_output * s * (1.0 - s)

    # 安全拷贝，避免原地副作用
    with torch.no_grad():
        grad_input.copy_(result)


def test(
    handle,
    device,
    shape,
    input_stride=None,
    grad_output_stride=None,
    grad_input_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    # 输入含正负值，便于覆盖 sigmoid 的不同区间
    input_tensor = TestTensor(
        shape, input_stride, dtype, device, mode="random", scale=4.0, bias=-2.0
    )
    grad_output = TestTensor(shape, grad_output_stride, dtype, device, mode="random")

    if inplace == Inplace.INPLACE_INPUT:
        if input_stride != grad_input_stride:
            return
        grad_input = input_tensor
    elif inplace == Inplace.INPLACE_GRAD_OUTPUT:
        if grad_input_stride != grad_output_stride:
            return
        grad_input = grad_output
    else:
        grad_input = TestTensor(shape, grad_input_stride, dtype, device, mode="zeros")

    if grad_input.is_broadcast():
        return

    print(
        f"Testing Sigmoid Backward on {InfiniDeviceNames[device]} with shape:{shape} "
        f"input_stride:{input_stride} grad_output_stride:{grad_output_stride} grad_input_stride:{grad_input_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    # 计算 PyTorch 参考结果（写入 grad_input.torch_tensor()）
    sigmoid_backward(
        grad_input.torch_tensor(),
        input_tensor.torch_tensor(),
        grad_output.torch_tensor(),
    )

    if sync is not None:
        sync()

    # 创建算子描述子
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSigmoidBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
            grad_input.descriptor,
            input_tensor.descriptor,
            grad_output.descriptor,
        )
    )

    # 使内部 desc 的 shape/stride 失效，强制 kernel 走外部传参
    for tensor in [input_tensor, grad_output, grad_input]:
        tensor.destroy_desc()

    # workspace
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
                input_tensor.data(),
                grad_output.data(),
                None,
            )
        )

    # 执行库实现，结果写入 grad_input.actual_tensor()
    lib_sigmoid_backward()

    # 校验
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(
            grad_input.actual_tensor(), grad_input.torch_tensor(), atol=atol, rtol=rtol
        )
    assert torch.allclose(
        grad_input.actual_tensor(), grad_input.torch_tensor(), atol=atol, rtol=rtol
    )

    # 性能分析（可选）
    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: sigmoid_backward(
                grad_input.torch_tensor(),
                input_tensor.torch_tensor(),
                grad_output.torch_tensor(),
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib",
            lambda: lib_sigmoid_backward(),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )

    check_error(LIBINFINIOP.infiniopDestroySigmoidBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # 覆盖运行选项
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mSigmoid Backward test passed!\033[0m")
