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
from libinfiniop import to_torch_dtype, torch_device_map
import numpy as np


# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # alpha, beta, a_shape, b_shape, c_shape, a_stride, b_stride, c_stride
    (1.0, 1.0, (16, 2048), (2048, 2048), (16, 2048), None, None, None),
    (1.0, 0.0, (2, 16, 2048), (2, 2048, 2048), (2, 16, 2048), None, None, None),
    (1.0, 1.0, (6, 2048), (2560, 2048), (6, 2560), None, None, None),
    (1.0 / 8.0, 0.0, (4, 8 * 6, 64), (4, 16, 64), (4, 8 * 6, 16), None, None, None),
]


# A B C D BIAS
# _TENSOR_DTYPES = [[InfiniDtype.F8E4M3, InfiniDtype.F8E4M3, InfiniDtype.F16, InfiniDtype.F16, InfiniDtype.F16]]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 0, "rtol": 5e-2},
    InfiniDtype.F8E4M3: {"atol": 0, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# PyTorch implementation for matrix multiplication


def linear_f8e4m3(_a, _b, _ans, scale_a, scale_b, alpha, bias, beta, _c):
    _a = _a.to(torch.float32)
    _a *= alpha
    _a = _a.to(torch.float8_e4m3fn)
    assert torch.cuda.get_device_capability() >= (9, 0)
    if len(_a.shape) > 2:
        for i in range(0, _a.shape[0]):
            torch._scaled_mm(
                _a[i],
                _b[i].T,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=bias,
                out_dtype=_ans.dtype,
                out=_ans[i],
            )
    else:
        torch._scaled_mm(
            _a,
            _b.T,
            scale_a=scale_a,
            scale_b=scale_b,
            bias=bias,
            out_dtype=_ans.dtype,
            out=_ans,
        )
    _ans += beta * _c


# Data types used for testing
# _TENSOR_DTYPES = [InfiniDtype.F8E4M3, InfiniDtype.F8E4M3, InfiniDtype.F8E4M3]
_TENSOR_DTYPES = [InfiniDtype.F8E4M3]

# A B C D BIAS
FP8_SUPPORT_COMBINES = [
    [
        InfiniDtype.F8E4M3,
        InfiniDtype.F8E4M3,
        InfiniDtype.F16,
        InfiniDtype.F16,
        InfiniDtype.F16,
    ],
    [
        InfiniDtype.F8E4M3,
        InfiniDtype.F8E4M3,
        InfiniDtype.BF16,
        InfiniDtype.BF16,
        InfiniDtype.BF16,
    ],
]


# The argument list should be (lib, handle, torch_device, <param list>, dtype)
# The <param list> should keep the same order as the one specified in _TEST_CASES
def test(
    handle,
    device,
    alpha,
    beta,
    a_shape,
    b_shape,
    c_shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=InfiniDtype.F16,
    sync=None,
):
    for precision in range(0, len(FP8_SUPPORT_COMBINES)):
        print(
            f"Testing LINEAR_FP8 on {InfiniDeviceNames[device]} with alpha:{alpha}, beta:{beta},"
            f" a_shape:{a_shape}, b_shape:{b_shape}, c_shape:{c_shape},"
            f" a_stride:{a_stride}, b_stride:{b_stride}, c_stride:{c_stride}, dtype:{InfiniDtypeNames[dtype]}"
            f" a_dtype:{InfiniDtypeNames[FP8_SUPPORT_COMBINES[precision][0]]}, b_dtype:{InfiniDtypeNames[FP8_SUPPORT_COMBINES[precision][1]]}"
            f" c_dtype:{InfiniDtypeNames[FP8_SUPPORT_COMBINES[precision][2]]}, d_dtype:{InfiniDtypeNames[FP8_SUPPORT_COMBINES[precision][3]]}"
        )

        # Initialize tensors
        a = TestTensor(a_shape, a_stride, InfiniDtype.F16, device)
        if a.dt != FP8_SUPPORT_COMBINES[precision][0]:
            a.convert_pricesion(FP8_SUPPORT_COMBINES[precision][0])
        b = TestTensor(b_shape, b_stride, InfiniDtype.F16, device)
        if b.dt != FP8_SUPPORT_COMBINES[precision][1]:
            b.convert_pricesion(FP8_SUPPORT_COMBINES[precision][1])
        c = TestTensor(
            c_shape, c_stride, FP8_SUPPORT_COMBINES[precision][2], device, mode="zeros"
        )
        d = TestTensor(
            c_shape, c_stride, FP8_SUPPORT_COMBINES[precision][3], device, mode="zeros"
        )
        ans = TestTensor(
            c_shape, c_stride, FP8_SUPPORT_COMBINES[precision][4], device, mode="zeros"
        )
        bias = TestTensor(
            (c_shape[-1],), None, FP8_SUPPORT_COMBINES[precision][2], device
        )

        scale_a = torch.tensor(1.0, device=torch_device_map[device])
        scale_b = torch.tensor(1.0, device=torch_device_map[device])

        def torch_linear():
            linear_f8e4m3(
                a.torch_tensor(),
                b.torch_tensor(),
                ans.torch_tensor(),
                scale_a=scale_a,
                scale_b=scale_b,
                alpha=alpha,
                bias=bias.torch_tensor(),
                beta=beta,
                _c=c.torch_tensor(),
            )

        torch_linear()

        if sync is not None:
            sync()

        descriptor = infiniopOperatorDescriptor_t()
        check_error(
            LIBINFINIOP.infiniopCreateLinearDescriptor(
                handle,
                ctypes.byref(descriptor),
                d.descriptor,
                a.descriptor,
                b.descriptor,
                c.descriptor,
            )
        )

        # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
        for tensor in [a, b, c]:
            tensor.destroy_desc()

        # Get workspace size and create workspace
        workspace_size = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetLinearWorkspaceSize(
                descriptor, ctypes.byref(workspace_size)
            )
        )
        workspace = TestWorkspace(workspace_size.value, device)

        scale_a_ = scale_a.clone()
        scale_b_ = scale_b.clone()

        # Execute infiniop gemm operator
        def lib_linear():
            check_error(
                LIBINFINIOP.infiniopLinear(
                    descriptor,
                    alpha,
                    a.data(),
                    scale_a_.data_ptr(),
                    b.data(),
                    scale_b_.data_ptr(),
                    beta,
                    c.data(),
                    None,
                    bias.data(),
                    d.data(),
                    None,
                    False,
                    False,
                    False,
                    workspace.data(),
                    workspace_size.value,
                    None,
                )
            )

        lib_linear()

        # Validate results
        atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

        if DEBUG:
            debug(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

        assert torch.allclose(
            d.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol
        )

        # Profiling workflow
        if PROFILE:
            # fmt: off
            profile_operation("PyTorch", lambda: torch_linear(), device, NUM_PRERUN, NUM_ITERATIONS)
            profile_operation("    lib", lambda: lib_linear(), device, NUM_PRERUN, NUM_ITERATIONS)
            # fmt: on
        check_error(LIBINFINIOP.infiniopDestroyLinearDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
