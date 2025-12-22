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

_TEST_CASES_ = [
    # shape, bias_exist, eps, input_strides, output_strides, weight_strides
    ((5, 4), True, 1e-5, None, None, None),
    ((5, 4, 32, 2048), True, 1e-5, None, None, None),
    ((13, 4, 4), True, 1e-5, [30, 4, 1], [50, 4, 1], [2]),
    ((16, 5, 563), True, 1e-4, None, None, None),
    ((5, 16, 563), False, 1e-5, None, None, [10]),
    ((4, 4, 563), True, 1e-5, None, None, None),
    ((40, 40, 56), True, 1e-5, [3600, 56, 1], None, None),
    ((40, 40, 56), False, 1e-5, [3600, 56, 1], None, None),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]


# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 5e-2, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 5e-2, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_layer_norm(
    output: torch.Tensor,
    output_standardization: torch.Tensor,
    output_rstd: torch.Tensor,
    input: torch.Tensor,
    weight,
    bias,
    eps,
    bias_exist: bool,
):
    original_dtype = input.dtype

    input_f32 = input.to(torch.float32)
    weight_f32 = weight.to(torch.float32)
    bias_f32 = bias.to(torch.float32) if bias_exist and bias is not None else None

    mean = input_f32.mean(dim=-1, keepdim=True)          # [..., 1]
    # print("mean in torch", mean)
    var = input_f32.var(dim=-1, keepdim=True, correction=0)  # [..., 1]
    # print("var in torch", var)

    rstd = torch.rsqrt(var + eps)  # [..., 1]
    # print("rstd in torch", rstd)
    centered_input = input_f32 - mean  # [..., D]
    normalized = centered_input * rstd  # [..., D]

    #  y = normalized * weight + bias
    output_f32 = normalized * weight_f32
    if bias_exist and bias_f32 is not None:
        output_f32 = output_f32 + bias_f32

    # 写回最终输出（转回原始 dtype）
    output.copy_(output_f32.to(original_dtype).detach())
    # 写入中间输出：centered input (x - μ)
    output_standardization.copy_(normalized.to(output_standardization.dtype).detach())
    # 写入中间输出：rstd，注意去掉最后的 keepdim 维度（从 [..., 1] → [...]）
    output_rstd.copy_(rstd.squeeze(-1).to(output_rstd.dtype).detach())


def test(
    handle,
    device,
    input_shape,
    bias_exist,
    eps,
    input_strides,
    output_strides,
    weight_strides,
    inplace,
    dtype,
    sync=None,
):
    print(
        f"Testing layer_norm on {InfiniDeviceNames[device]} with input_shape:{input_shape},"
        f"bias:{bias_exist},eps:{eps},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    input_standardization = TestTensor(
        input_shape,
        None,
        dtype,
        device,
    )

    input_std_deviation = TestTensor(
        input_shape[:-1],
        None,
        dtype,
        device,
    )

    input = TestTensor(input_shape, input_strides, dtype, device)
    if inplace == Inplace.INPLACE:
        if output_strides != input_strides:
            return
        output = input
    else:
        output = TestTensor(
            input_shape,
            output_strides,
            dtype,
            device,
        )

    weight = TestTensor(
        input_shape[-1:],
        weight_strides,
        dtype,
        device,
    )

    bias = (
        TestTensor(
            input_shape[-1:],
            None,
            dtype,
            device,
        )
        if bias_exist
        else None
    )

    torch_layer_norm(
        output.torch_tensor(),
        input_standardization.torch_tensor(),
        input_std_deviation.torch_tensor(),
        input.torch_tensor(),
        weight.torch_tensor(),
        bias.torch_tensor() if bias_exist else None,
        eps,
        bias_exist,
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateLayerNormDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            input_standardization.descriptor,
            input_std_deviation.descriptor,
            input.descriptor,
            weight.descriptor,
            bias.descriptor if bias_exist else None,
            eps,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in (
        [output, input_standardization, input_std_deviation, input, weight] + [bias]
        if bias_exist
        else []
    ):
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLayerNormWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output.device)

    def lib_layer_norm():
        check_error(
            LIBINFINIOP.infiniopLayerNorm(
                descriptor,
                workspace.data(),
                workspace.size(),
                output.data(),
                input_standardization.data(),
                input_std_deviation.data(),
                input.data(),
                weight.data(),
                bias.data() if bias_exist else None,
                None,
            )
        )

    lib_layer_norm()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)
        debug(
            input_standardization.actual_tensor(),
            input_standardization.torch_tensor(),
            atol=atol,
            rtol=rtol,
        )
        debug(
            input_std_deviation.actual_tensor(),
            input_std_deviation.torch_tensor(),
            atol=atol,
            rtol=rtol,
        )

    assert torch.allclose(
        output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol
    )
    assert torch.allclose(
        input_standardization.actual_tensor(),
        input_standardization.torch_tensor(),
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(
        input_std_deviation.actual_tensor(),
        input_std_deviation.torch_tensor(),
        atol=atol,
        rtol=rtol,
    )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_layer_norm(
            output, input_standardization, input_std_deviation, input, weight, bias, eps, bias_exist
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_layer_norm(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyLayerNormDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest layer_norm passed!\033[0m")
