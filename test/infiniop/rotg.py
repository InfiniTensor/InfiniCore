import ctypes
import math
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
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    test_operator,
)

_TEST_CASES = [
    (0.0, 0.0),
    (3.0, 4.0),
    (-2.5, 5.0),
    (7.0, -1.5),
    (-3.2, -8.4),
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
    InfiniDtype.F64: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}


def torch_rotg(a, b):
    anorm = abs(a)
    bnorm = abs(b)
    if bnorm == 0.0:
        return a, 0.0, 1.0, 0.0
    if anorm == 0.0:
        return b, 1.0, 0.0, 1.0

    sigma = math.copysign(1.0, a if anorm > bnorm else b)
    r = sigma * math.hypot(a, b)
    c = a / r
    s = b / r
    if anorm > bnorm:
        z = s
    elif c != 0.0:
        z = 1.0 / c
    else:
        z = 1.0
    return r, z, c, s


def test(handle, device, a0, b0, dtype=torch.float32, sync=None):
    a_torch = torch.tensor([a0])
    b_torch = torch.tensor([b0])
    a = TestTensor(
        a_torch.shape,
        a_torch.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=a_torch,
    )
    b = TestTensor(
        b_torch.shape,
        b_torch.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=b_torch,
    )
    c = TestTensor(tuple(), None, dtype, device, mode="zeros")
    s = TestTensor(tuple(), None, dtype, device, mode="zeros")

    exp_a, exp_b, exp_c, exp_s = torch_rotg(a0, b0)

    print(
        f"Testing Rotg on {InfiniDeviceNames[device]} with a:{a0} b:{b0} dtype:{InfiniDtypeNames[dtype]}"
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRotgDescriptor(
            handle,
            ctypes.byref(descriptor),
            a.descriptor,
            b.descriptor,
            c.descriptor,
            s.descriptor,
        )
    )

    a.destroy_desc()
    b.destroy_desc()
    c.destroy_desc()
    s.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRotgWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    check_error(
        LIBINFINIOP.infiniopRotg(
            descriptor,
            workspace.data(),
            workspace.size(),
            a.data(),
            b.data(),
            c.data(),
            s.data(),
            None,
        )
    )

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    assert math.isclose(a.actual_tensor().item(), exp_a, rel_tol=rtol, abs_tol=atol)
    assert math.isclose(b.actual_tensor().item(), exp_b, rel_tol=rtol, abs_tol=atol)
    assert math.isclose(c.actual_tensor().item(), exp_c, rel_tol=rtol, abs_tol=atol)
    assert math.isclose(s.actual_tensor().item(), exp_s, rel_tol=rtol, abs_tol=atol)

    check_error(LIBINFINIOP.infiniopDestroyRotgDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")
