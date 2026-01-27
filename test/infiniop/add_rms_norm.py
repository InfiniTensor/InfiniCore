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

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # y_shape, a_shape, b_shape, w_shape, y_stride, a_stride, b_stride
    ((1, 4), (1, 4), (1, 4), (4,), None, None, None),
    ((2, 4), (2, 4), (2, 4), (4,), None, None, None),
    ((2, 2, 4), (2, 2, 4), (2, 2, 4), (4,), None, None, None),
    ((2, 2, 4), (2, 2, 4), (2, 2, 4), (4,), (12, 8, 1), (12, 8, 1), (12, 8, 1)),
    ((16, 2048), (16, 2048), (16, 2048), (2048,), None, None, None),
    ((16, 2048), (16, 2048), (16, 2048), (2048,), (4096, 1), (4096, 1), (4096, 1)),
    ((15, 3584), (15, 3584), (15, 3584), (3584,), None, None, None),
    ((4, 4, 2048), (4, 4, 2048), (4, 4, 2048), (2048,), None, None, None),
    (
        (4, 4, 2048),
        (4, 4, 2048),
        (4, 4, 2048),
        (2048,),
        (2048, 8192, 1),
        (2048, 8192, 1),
        (2048, 8192, 1),
    ),
    (
        (4, 4, 2048),
        (4, 4, 2048),
        (4, 4, 2048),
        (2048,),
        (16384, 4096, 1),
        (16384, 4096, 1),
        (16384, 4096, 1),
    ),
    ((15, 3584), (15, 3584), (15, 3584), (3584,), None, None, None),
    ((15, 8192), (15, 8192), (15, 8192), (8192,), None, None, None),
]

# w (weight) types
# Note: 'None' means the same as input dtype
_WEIGHT_DTYPES = [None, InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16]
# a, b types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]

# Form the test cases by appending each element of _WEIGHT_DTYPES to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (w_dtype,) for test_case in _TEST_CASES_ for w_dtype in _WEIGHT_DTYPES
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 2e-3, "rtol": 2e-3},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def add_rms_norm(ans, a, b, w, eps):
    input_dtype = a.dtype
    # Compute add(a, b)
    sum_tensor = a.to(torch.float32) + b.to(torch.float32)
    # Compute RMS normalization
    scale = sum_tensor.pow(2).mean(-1, keepdim=True).add_(eps).rsqrt_()
    ans.set_((sum_tensor.mul_(scale).mul_(w.to(torch.float32))).to(input_dtype))


def test(
    handle,
    device,
    y_shape,
    a_shape,
    b_shape,
    w_shape,
    y_stride,
    a_stride,
    b_stride,
    w_dtype=InfiniDtype.F32,
    dtype=InfiniDtype.F16,
    sync=None,
):
    w_dtype = w_dtype if w_dtype else dtype
    print(
        f"Testing AddRMSNorm on {InfiniDeviceNames[device]} with y_shape:{y_shape} a_shape:{a_shape} b_shape:{b_shape} w_shape:{w_shape}"
        f" y_stride:{y_stride} a_stride:{a_stride} b_stride:{b_stride} w_dtype:{InfiniDtypeNames[w_dtype]} dtype:{InfiniDtypeNames[dtype]}"
    )

    y = TestTensor(y_shape, y_stride, dtype, device, mode="ones")
    residual_out = TestTensor(a_shape, a_stride, dtype, device, mode="ones")
    a = TestTensor(a_shape, a_stride, dtype, device, scale=0.01)
    b = TestTensor(b_shape, b_stride, dtype, device, scale=0.01)
    w = TestTensor(w_shape, None, w_dtype, device)

    eps = 1e-6
    add_rms_norm(
        y.torch_tensor(), a.torch_tensor(), b.torch_tensor(), w.torch_tensor(), eps
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()

    check_error(
        LIBINFINIOP.infiniopCreateAddRMSNormDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            residual_out.descriptor,
            a.descriptor,
            b.descriptor,
            w.descriptor,
            eps,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a, b, y, w, residual_out]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAddRMSNormWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_add_rms_norm():
        check_error(
            LIBINFINIOP.infiniopAddRMSNorm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                residual_out.data(),
                a.data(),
                b.data(),
                w.data(),
                None,
            )
        )

    lib_add_rms_norm()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    # Verify normalized result (y)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Verify add result (residual_out) - should be a + b
    expected_residual = a.torch_tensor().to(torch.float32) + b.torch_tensor().to(
        torch.float32
    )
    expected_residual = expected_residual.to(a.torch_tensor().dtype)
    if DEBUG:
        debug(residual_out.actual_tensor(), expected_residual, atol=atol, rtol=rtol)
    assert torch.allclose(
        residual_out.actual_tensor(), expected_residual, atol=atol, rtol=rtol
    )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: add_rms_norm(y.torch_tensor(), a.torch_tensor(), b.torch_tensor(), w.torch_tensor(), eps), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_add_rms_norm(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyAddRMSNormDescriptor(descriptor))


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
