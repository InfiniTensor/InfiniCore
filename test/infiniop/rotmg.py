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
    (1.0, 2.0, 3.0, 4.0),
    (2.5, 0.5, -1.2, 0.8),
    (3.0, 4.0, 0.0, 2.0),
    (1.5, 1.5, 2.0, -3.0),
]

_TENSOR_DTYPES = [
    # InfiniDtype.F16,
    InfiniDtype.F32,
    # InfiniDtype.F64,
    # InfiniDtype.BF16,
]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F64: {"atol": 1e-12, "rtol": 1e-12},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}


def _torch_dtype(dtype):
    if dtype == InfiniDtype.F64:
        return torch.float64
    return torch.float32


def torch_rotmg(d1, d2, x1, y1):
    zero = 0.0
    one = 1.0
    two = 2.0
    gam = 4096.0
    gamsq = 1.67772e7
    rgamsq = 5.96046e-8

    sparam = [0.0] * 5
    sh11 = sh12 = sh21 = sh22 = 0.0

    if d1 < zero:
        sflag = -one
        d1 = d2 = x1 = zero
    else:
        sp2 = d2 * y1
        if sp2 == zero:
            sparam[0] = -two
            return d1, d2, x1, sparam

        sp1 = d1 * x1
        sq2 = sp2 * y1
        sq1 = sp1 * x1

        if abs(sq1) > abs(sq2):
            sh21 = -y1 / x1
            sh12 = sp2 / sp1
            su = one - sh12 * sh21
            if su > zero:
                sflag = zero
                d1 = d1 / su
                d2 = d2 / su
                x1 = x1 * su
            else:
                sflag = -one
                sh11 = sh12 = sh21 = sh22 = zero
                d1 = d2 = x1 = zero
        else:
            if sq2 < zero:
                sflag = -one
                d1 = d2 = x1 = zero
            else:
                sflag = one
                sh11 = sp1 / sp2
                sh22 = x1 / y1
                su = one + sh11 * sh22
                stemp = d2 / su
                d2 = d1 / su
                d1 = stemp
                x1 = y1 * su

        if d1 != zero:
            while d1 <= rgamsq or d1 >= gamsq:
                if sflag == zero:
                    sh11 = one
                    sh22 = one
                    sflag = -one
                else:
                    sh21 = -one
                    sh12 = one
                    sflag = -one
                if d1 <= rgamsq:
                    d1 = d1 * gam * gam
                    x1 = x1 / gam
                    sh11 = sh11 / gam
                    sh12 = sh12 / gam
                else:
                    d1 = d1 / (gam * gam)
                    x1 = x1 * gam
                    sh11 = sh11 * gam
                    sh12 = sh12 * gam

        if d2 != zero:
            while abs(d2) <= rgamsq or abs(d2) >= gamsq:
                if sflag == zero:
                    sh11 = one
                    sh22 = one
                    sflag = -one
                else:
                    sh21 = -one
                    sh12 = one
                    sflag = -one
                if abs(d2) <= rgamsq:
                    d2 = d2 * gam * gam
                    sh21 = sh21 / gam
                    sh22 = sh22 / gam
                else:
                    d2 = d2 / (gam * gam)
                    sh21 = sh21 * gam
                    sh22 = sh22 * gam

    if sflag < zero:
        sparam[1] = sh11
        sparam[2] = sh21
        sparam[3] = sh12
        sparam[4] = sh22
    elif sflag == zero:
        sparam[2] = sh21
        sparam[3] = sh12
    else:
        sparam[1] = sh11
        sparam[4] = sh22

    sparam[0] = sflag
    return d1, d2, x1, sparam


def test(handle, device, d1_0, d2_0, x1_0, y1_0, dtype=torch.float32, sync=None):
    exp_d1, exp_d2, exp_x1, exp_sparam = torch_rotmg(d1_0, d2_0, x1_0, y1_0)

    scalar_dtype = _torch_dtype(dtype)
    d1_torch = torch.tensor([d1_0], dtype=scalar_dtype)
    d2_torch = torch.tensor([d2_0], dtype=scalar_dtype)
    x1_torch = torch.tensor([x1_0], dtype=scalar_dtype)
    y1_torch = torch.tensor([y1_0], dtype=scalar_dtype)
    d1 = TestTensor(
        d1_torch.shape,
        d1_torch.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=d1_torch,
    )
    d2 = TestTensor(
        d2_torch.shape,
        d2_torch.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=d2_torch,
    )
    x1 = TestTensor(
        x1_torch.shape,
        x1_torch.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=x1_torch,
    )
    y1 = TestTensor(
        y1_torch.shape,
        y1_torch.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=y1_torch,
    )
    param = TestTensor((5,), (1,), dtype, device, mode="zeros")

    print(
        f"Testing Rotmg on {InfiniDeviceNames[device]} with d1:{d1_0} d2:{d2_0} x1:{x1_0} y1:{y1_0} dtype:{InfiniDtypeNames[dtype]}"
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRotmgDescriptor(
            handle,
            ctypes.byref(descriptor),
            d1.descriptor,
            d2.descriptor,
            x1.descriptor,
            y1.descriptor,
            param.descriptor,
        )
    )

    d1.destroy_desc()
    d2.destroy_desc()
    x1.destroy_desc()
    y1.destroy_desc()
    param.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRotmgWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    check_error(
        LIBINFINIOP.infiniopRotmg(
            descriptor,
            workspace.data(),
            workspace.size(),
            d1.data(),
            d2.data(),
            x1.data(),
            y1.data(),
            param.data(),
            None,
        )
    )

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    assert math.isclose(d1.actual_tensor().item(), exp_d1, rel_tol=rtol, abs_tol=atol)
    assert math.isclose(d2.actual_tensor().item(), exp_d2, rel_tol=rtol, abs_tol=atol)
    assert math.isclose(x1.actual_tensor().item(), exp_x1, rel_tol=rtol, abs_tol=atol)
    for i in range(5):
        assert math.isclose(
            param.actual_tensor()[i].item(), exp_sparam[i], rel_tol=rtol, abs_tol=atol
        )

    check_error(LIBINFINIOP.infiniopDestroyRotmgDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")
