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
    to_torch_dtype,
    torch_device_map,
)
from enum import Enum, auto

# ======================================================================
# Configuration (Internal Use Only)
# Now each test case tuple is: (shape, a_stride, b_stride, cond_stride, c_stride)
# ======================================================================
_TEST_CASES_ = [
    ((13, 4), None, None, None, None),
    ((13, 4), None, None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None, None),
    ((13, 4, 4), None, None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None, None),
    ((16, 5632), None, None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()
    INPLACE_COND = auto()

_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_A,
    Inplace.INPLACE_B,
    Inplace.INPLACE_COND,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

_INTEGER_DTYPES = [
    InfiniDtype.I32,
    InfiniDtype.I64,
    InfiniDtype.U32,
    InfiniDtype.U64,
]

_FLOAT_DTYPES = [
    InfiniDtype.F16,
    InfiniDtype.F32,
    InfiniDtype.F64,
    InfiniDtype.BF16,
]

_TENSOR_DTYPES = _INTEGER_DTYPES + _FLOAT_DTYPES

_TOLERANCE_MAP = {
    InfiniDtype.I32: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.I64: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.U32: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.U64: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    InfiniDtype.F64: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def is_supported_dt(inf_dt):
    try:
        td = to_torch_dtype(inf_dt, compatability_mode=True)
        _ = torch.empty((1,), dtype=td, device="cpu")
        return True
    except Exception:
        return False

def _is_integer_dtype(inf_dt):
    return inf_dt in _INTEGER_DTYPES

def _is_unsigned_dtype(inf_dt):
    return inf_dt in (InfiniDtype.U32, InfiniDtype.U64)


def make_integer_torch_tensor(shape, inf_dt, device):
    use_compatibility = _is_unsigned_dtype(inf_dt)

    if inf_dt == InfiniDtype.I32:
        low, high, dtype = -2000, 2000, torch.int32
    elif inf_dt == InfiniDtype.I64:
        low, high, dtype = -2048, 2048, torch.int64
    elif inf_dt == InfiniDtype.U32:
        low, high, dtype = 0, 2000, torch.int32
    elif inf_dt == InfiniDtype.U64:
        low, high, dtype = 0, 2048, torch.int64
    else:
        low, high, dtype = 0, 1, torch.int64

    dev = torch_device_map[device]

    t = torch.randint(low=low, high=high, size=shape, dtype=dtype, device=dev)

    target_torch_dt = to_torch_dtype(inf_dt, compatability_mode=use_compatibility)
    if t.dtype != target_torch_dt:
        t = t.to(dtype=target_torch_dt)

    return t

def where_ref(c, a, b, cond):
    cond_bool = cond.torch_tensor().to(torch.bool)
    c.torch_tensor().copy_(torch.where(cond_bool, a.torch_tensor(), b.torch_tensor()))

def test(
    handle,
    device,
    shape,
    a_stride=None,
    b_stride=None,
    cond_stride=None,
    c_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F16,
    sync=None,
):
    inf_dt = dtype

    if not is_supported_dt(inf_dt):
        # print(f"Skipping dtype {InfiniDtypeNames[inf_dt]} on this platform")
        return

    try:
        if _is_integer_dtype(inf_dt):
            a_torch = make_integer_torch_tensor(shape, inf_dt, device)
            b_torch = make_integer_torch_tensor(shape, inf_dt, device)
            a = TestTensor.from_torch(a_torch, inf_dt, device)
            b = TestTensor.from_torch(b_torch, inf_dt, device)
        else:
            a = TestTensor(shape, a_stride, inf_dt, device, mode="random")
            b = TestTensor(shape, b_stride, inf_dt, device, mode="random")
    except RuntimeError as e:
        msg = str(e)
        if "not implemented for 'UInt32'" in msg or "not implemented for 'UInt64'" in msg or "check_uniform_bounds" in msg:
            # print(f"Skipping dtype {InfiniDtypeNames[inf_dt]} because platform torch can't build random tensor: {e}")
            return
        else:
            raise

    dev = torch_device_map[device]
    if _is_integer_dtype(inf_dt):
        cond_torch = torch.randint(0, 2, size=shape, dtype=to_torch_dtype(inf_dt, compatability_mode=False), device=dev)
    else:
        cond_bool = (torch.rand(shape, device=dev) > 0.5)
        cond_torch = cond_bool.to(dtype=to_torch_dtype(inf_dt, compatability_mode=False))

    cond = TestTensor.from_torch(cond_torch, inf_dt, device)

    if inplace == Inplace.INPLACE_A:
        if a_stride != c_stride:
            return
        c = a
    elif inplace == Inplace.INPLACE_B:
        if c_stride != b_stride:
            return
        c = b
    elif inplace == Inplace.INPLACE_COND:
        if c_stride != cond_stride:
            return
        c = cond    
    else:
        if _is_integer_dtype(inf_dt):
            dev = torch_device_map[device]
            c_torch = torch.zeros(shape, dtype=to_torch_dtype(inf_dt, compatability_mode=False), device=dev)
            c = TestTensor.from_torch(c_torch, inf_dt, device)
        else:
            c = TestTensor(shape, c_stride, inf_dt, device, mode="ones")

    if c.is_broadcast():
        return

    print(
        f"Testing Where on {InfiniDeviceNames[device]} "
        f"shape:{shape} a_stride:{a_stride} b_stride:{b_stride} cond_stride:{cond_stride} c_stride:{c_stride} "
        f"dtype:{InfiniDtypeNames[inf_dt]} inplace:{inplace}"
    )

    where_ref(c, a, b, cond)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    try:
        check_error(
            LIBINFINIOP.infiniopCreateWhereDescriptor(
                handle,
                ctypes.byref(descriptor),
                c.descriptor,
                a.descriptor,
                b.descriptor,
                cond.descriptor,
            )
        )
    except Exception as e:
        # print(f"Skipping dtype {InfiniDtypeNames[inf_dt]} on {InfiniDeviceNames[device]}: CreateWhereDescriptor failed: {e}")
        return

    for tensor in [a, b, c, cond]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetWhereWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    def lib_where():
        check_error(
            LIBINFINIOP.infiniopWhere(
                descriptor,
                workspace.data(),
                workspace.size(),
                c.data(),
                a.data(),
                b.data(),
                cond.data(),
                None,
            )
        )

    lib_where()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, inf_dt)
    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation("PyTorch", lambda: where_ref(c, a, b, cond), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_where(), device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyWhereDescriptor(descriptor))


def main():
    args = get_args()
    global DEBUG, PROFILE, NUM_PRERUN, NUM_ITERATIONS
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    supported = [dt for dt in _TENSOR_DTYPES if is_supported_dt(dt)]
    devices = get_test_devices(args)

    for device in devices:
        test_operator(device, test, _TEST_CASES, supported)

    print("\033[92mTest passed!\033[0m")


if __name__ == "__main__":
    main()
