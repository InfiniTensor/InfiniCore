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
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
    to_torch_dtype,
    torch_device_map
)
import itertools

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
_TEST_CASES = [
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None),
    ((13, 4, 4), None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), None),
    ((16, 5632), None, None),
    ((16, 5632), (10240, 1), (10240, 1)),
    ((4, 4, 5632), None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
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
]

def is_supported_dt(inf_dt):
    try:
        td = to_torch_dtype(inf_dt, compatability_mode=True)
        _ = torch.empty((1,), dtype=td, device="cpu")
        return True
    except Exception:
        return False

_TOLERANCE_MAP = {
    ("float", "float"): {"atol": 1e-3, "rtol": 1e-3},
    ("int", "float"): {"atol": 1.0, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def _is_integer_dtype(inf_dt):
    return inf_dt in _INTEGER_DTYPES


def _is_float_dtype(inf_dt):
    return inf_dt in _FLOAT_DTYPES


def _is_unsigned_dtype(inf_dt):
    return inf_dt in (InfiniDtype.U32, InfiniDtype.U64)


def reference_cast_torch(output_tensor, input_tensor):
    converted = input_tensor.to(dtype=output_tensor.dtype, device=output_tensor.device).clone()
    output_tensor.copy_(converted)


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


def test(
    handle,
    device,
    shape,
    in_stride,
    out_stride,
    dtype_pair,
    sync=None, 
):
    in_dt, out_dt = dtype_pair

    if not is_supported_dt(in_dt) or not is_supported_dt(out_dt):
        print(f"Skipping test for in={InfiniDtypeNames[in_dt]} out={InfiniDtypeNames[out_dt]} because dtype not supported on this platform")
        return

    try:
        if _is_integer_dtype(in_dt):
            in_torch = make_integer_torch_tensor(shape, in_dt, device)
            input = TestTensor.from_torch(in_torch, in_dt, device)
        else:
            input = TestTensor(shape, in_stride, in_dt, device, mode="random")

        output = TestTensor(shape, out_stride, out_dt, device, mode="zeros")

        if output.is_broadcast():
            return

        print(f"Testing Cast on {InfiniDeviceNames[device]} shape={shape} in={InfiniDtypeNames[in_dt]} out={InfiniDtypeNames[out_dt]} in_stride={in_stride} out_stride={out_stride}")

        reference_cast_torch(output.actual_tensor(), input.torch_tensor())

        expected = output.actual_tensor().clone()

        descriptor = infiniopOperatorDescriptor_t()
        check_error(
            LIBINFINIOP.infiniopCreateCastDescriptor(
                handle,
                ctypes.byref(descriptor),
                output.descriptor,
                input.descriptor,
            )
        )

        input.destroy_desc()
        output.destroy_desc()

        workspace_size = c_uint64(0)
        check_error(LIBINFINIOP.infiniopGetCastWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
        workspace = TestWorkspace(workspace_size.value, device)

        def lib_cast():
            check_error(
                LIBINFINIOP.infiniopCast(
                    descriptor,
                    workspace.data(),
                    workspace_size.value,
                    output.data(),
                    input.data(),
                    None,
                )
            )

        lib_cast()

        actual = output.actual_tensor()

        if _is_integer_dtype(in_dt) and _is_float_dtype(out_dt):
            tol = _TOLERANCE_MAP[("int", "float")]
            atol, rtol = tol["atol"], tol["rtol"]
        elif _is_float_dtype(in_dt) and _is_float_dtype(out_dt):
            tol = _TOLERANCE_MAP[("float", "float")]
            atol, rtol = tol["atol"], tol["rtol"]
        else:
            atol, rtol = 0, 0

        if DEBUG:
            debug(actual, expected, atol=atol, rtol=rtol)

        assert torch.allclose(actual, expected, atol=atol, rtol=rtol), \
            f"Mismatch for in={InfiniDtypeNames[in_dt]} out={InfiniDtypeNames[out_dt]} shape={shape}"

        if PROFILE:
            profile_operation("PyTorch", lambda: reference_cast_torch(output.torch_tensor(), input.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
            profile_operation("    lib", lambda: lib_cast(), device, NUM_PRERUN, NUM_ITERATIONS)

        check_error(LIBINFINIOP.infiniopDestroyCastDescriptor(descriptor))
        
    except RuntimeError as e:
        if "not implemented for 'UInt32'" in str(e) or "not implemented for 'UInt64'" in str(e):
            #print(f"Skipping unsupported operation: {e}")
            return False
        else:
            raise


def main():
    args = get_args()
    global DEBUG, PROFILE, NUM_PRERUN, NUM_ITERATIONS
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    integer_pairs = itertools.product(_INTEGER_DTYPES, _INTEGER_DTYPES)
    float_pairs = itertools.product(_FLOAT_DTYPES, _FLOAT_DTYPES)
    int_to_float_pairs = itertools.product(_INTEGER_DTYPES, _FLOAT_DTYPES)

    all_pairs = list(set(itertools.chain(integer_pairs, float_pairs, int_to_float_pairs)))

    supported_pairs = []
    skipped_pairs = []
    for pair in all_pairs:
        in_dt, out_dt = pair
        if is_supported_dt(in_dt) and is_supported_dt(out_dt):
            supported_pairs.append(pair)
        else:
            skipped_pairs.append(pair)

    print(f"Supported dtype pairs: {[(InfiniDtypeNames[in_d], InfiniDtypeNames[out_d]) for in_d, out_d in supported_pairs]}")
    if skipped_pairs:
        print(f"Warning: skipping unsupported dtype pairs: {[(InfiniDtypeNames[in_d], InfiniDtypeNames[out_d]) for in_d, out_d in skipped_pairs]}")

    devices = get_test_devices(args)

    for device in devices:
        test_operator(device, test, _TEST_CASES, supported_pairs)

    print("\033[92mAll cast tests passed!\033[0m")


if __name__ == "__main__":
    main()
