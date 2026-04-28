import ctypes
from ctypes import c_uint64

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
)

_TEST_CASES = [
    ((13,), None, None, (-1.0, 1.2, -0.3, 0.4, 0.8)),
    ((13,), (10,), (10,), (0.0, 0.0, -0.25, 0.5, 0.0)),
    ((5632,), None, None, (1.0, 1.1, 0.0, 0.0, 0.9)),
    ((5632,), (5,), (5,), (-2.0, 0.0, 0.0, 0.0, 0.0)),
]

_TENSOR_DTYPES = [
    # InfiniDtype.F16,
    InfiniDtype.F32,
    # InfiniDtype.F64,
    # InfiniDtype.BF16,
]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.F64: {"atol": 1e-9, "rtol": 1e-9},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_rotm(x, y, param):
    sflag, sh11, sh21, sh12, sh22 = param
    if sflag == -2.0:
        return

    w = x.clone()
    z = y.clone()

    if sflag < 0.0:
        x.copy_(w * sh11 + z * sh12)
        y.copy_(w * sh21 + z * sh22)
    elif sflag == 0.0:
        x.copy_(w + z * sh12)
        y.copy_(w * sh21 + z)
    else:
        x.copy_(w * sh11 + z)
        y.copy_(-w + sh22 * z)


def _torch_dtype(dtype):
    if dtype == InfiniDtype.F64:
        return torch.float64
    return torch.float32


def test(
    handle,
    device,
    shape,
    x_stride=None,
    y_stride=None,
    param=(-1.0, 1.2, -0.3, 0.4, 0.8),
    dtype=torch.float32,
    sync=None,
):
    x = TestTensor(shape, x_stride, dtype, device)
    y = TestTensor(shape, y_stride, dtype, device)
    param_tensor = TestTensor(
        (5,),
        (1,),
        dtype,
        device,
        mode="manual",
        set_tensor=torch.tensor(param, dtype=_torch_dtype(dtype)),
    )

    if x.is_broadcast() or y.is_broadcast():
        return

    print(
        f"Testing Rotm on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} "
        f"y_stride:{y_stride} param:{param} dtype:{InfiniDtypeNames[dtype]}"
    )

    torch_rotm(x.torch_tensor(), y.torch_tensor(), param)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRotmDescriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor,
            y.descriptor,
            param_tensor.descriptor,
        )
    )

    x.destroy_desc()
    y.destroy_desc()
    param_tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRotmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_rotm():
        check_error(
            LIBINFINIOP.infiniopRotm(
                descriptor,
                workspace.data(),
                workspace.size(),
                x.data(),
                y.data(),
                param_tensor.data(),
                None,
            )
        )

    lib_rotm()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(x.actual_tensor(), x.torch_tensor(), atol=atol, rtol=rtol)
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(x.actual_tensor(), x.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: torch_rotm(x.torch_tensor(), y.torch_tensor(), param),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_rotm(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyRotmDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
