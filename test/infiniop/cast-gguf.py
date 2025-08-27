import torch
import ctypes
from ctypes import c_uint64
from enum import Enum, auto
from gguf import GGUFReader

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

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()

FLOAT_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.F64]
INTER_DTYPES = [InfiniDtype.I32, InfiniDtype.I64, InfiniDtype.U32, InfiniDtype.U64]

_TENSOR_DTYPES = [
    (ftype, ttype) for ftype in FLOAT_DTYPES + INTER_DTYPES for ttype in FLOAT_DTYPES
] + [
    (ftype, ttype) for ftype in INTER_DTYPES for ttype in INTER_DTYPES
]

_TOLERANCE_MAP = {
    (ftype, ttype): {"atol": 1e-3, "rtol": 1e-3}
    for ftype in FLOAT_DTYPES + INTER_DTYPES
    for ttype in FLOAT_DTYPES
}
_TOLERANCE_MAP.update({
    (ftype, ttype): {"atol": 0, "rtol": 0}
    for ftype in INTER_DTYPES
    for ttype in INTER_DTYPES
})
_TOLERANCE_MAP.update({
    (InfiniDtype.F64, InfiniDtype.F16): {"atol": 1e-3, "rtol": 1e-3},
    (InfiniDtype.I16, InfiniDtype.I32): {"atol": 1e-3, "rtol": 1e-3}
})

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def cast(c: torch.Tensor, x: torch.Tensor):
    if not x.device.type.startswith('cpu') and c.dtype in [torch.uint32, torch.uint64]:
        x_np = x.cpu().numpy()
        if c.dtype == torch.uint32:
            c_np = x_np.astype('uint32')
        elif c.dtype == torch.uint64:
            c_np = x_np.astype('uint64')
        c.copy_(torch.from_numpy(c_np))
    else:
        c.copy_(x.to(c.dtype))

def load_test_cases_from_gguf(filepath):
    reader = GGUFReader(filepath)
    tensors = reader.tensors

    test_cases = []
    for tensor in tensors:
        data = tensor.data
        shape = data.shape
        torch_tensor = torch.from_numpy(data.copy())
        x_stride = torch_tensor.stride()
        c_stride = None

        for inplace in [Inplace.OUT_OF_PLACE, Inplace.INPLACE_X]:
            test_cases.append((shape, x_stride, c_stride, inplace, torch_tensor))

    return test_cases

def test(
    handle,
    device,
    shape,
    x_stride=None,
    c_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    torch_tensor=None,
    dtype=(InfiniDtype.F32, InfiniDtype.F64),
    sync=None,
):
    x = TestTensor(shape, x_stride, dtype[0], device, mode="manual", set_tensor=torch_tensor)
    if inplace == Inplace.INPLACE_X:
        if x_stride != c_stride:
            return
        c = x
    else:
        c = TestTensor(shape, c_stride, dtype[1], device)

    print(
        f"Testing Cast on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} c_stride:{c_stride} "
        f"dtype from {InfiniDtypeNames[dtype[0]]} to {InfiniDtypeNames[dtype[1]]} inplace={inplace}"
    )

    cast(c.torch_tensor(), x.torch_tensor())
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateCastDescriptor(
            handle, ctypes.byref(descriptor), c.descriptor, x.descriptor
        )
    )

    for tensor in [x, c]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetCastWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    def lib_cast():
        check_error(
            LIBINFINIOP.infiniopCast(
                descriptor,
                workspace.data(),
                workspace.size(),
                c.data(),
                x.data(),
                None
            )
        )

    lib_cast()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    actual = c.actual_tensor()
    expected = c.torch_tensor()

    if DEBUG:
        debug(actual, expected, atol=atol, rtol=rtol)

    if expected.dtype in [torch.float16, torch.float32, torch.float64]:
        assert torch.allclose(actual, expected, atol=atol, rtol=rtol)
    else:
        assert torch.equal(actual, expected), f"Integer cast mismatch!\nExpected:\n{expected}\nActual:\n{actual}"

    if PROFILE:
        profile_operation("PyTorch", lambda: cast(c.torch_tensor(), x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_cast, device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyCastDescriptor(descriptor))

if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # gguf 文件路径你可以根据实际改
    _TEST_CASES = {
        InfiniDtype.F16: load_test_cases_from_gguf("T1-1-1/cast/cast_f16.gguf"),
        InfiniDtype.F32: load_test_cases_from_gguf("T1-1-1/cast/cast_f32.gguf"),
        InfiniDtype.F64: load_test_cases_from_gguf("T1-1-1/cast/cast_float64.gguf"),
        InfiniDtype.I32: load_test_cases_from_gguf("T1-1-1/cast/cast_i32.gguf"),
        InfiniDtype.I64: load_test_cases_from_gguf("T1-1-1/cast/cast_i64.gguf"),
        InfiniDtype.U32: load_test_cases_from_gguf("T1-1-1/cast/cast_u32.gguf"),
        InfiniDtype.U64: load_test_cases_from_gguf("T1-1-1/cast/cast_u64.gguf"),
    }

    for device in get_test_devices(args):
        for ftype in _TEST_CASES:
            if ftype in FLOAT_DTYPES:
                test_operator(device, test, _TEST_CASES[ftype], [(f, t) for f in FLOAT_DTYPES for t in FLOAT_DTYPES])
            else:
                test_operator(device, test, _TEST_CASES[ftype], [(f, t) for f in INTER_DTYPES for t in FLOAT_DTYPES+INTER_DTYPES if t not in [InfiniDtype.U64, InfiniDtype.U32] and f not in [InfiniDtype.U64, InfiniDtype.U32]])
                

    print("\033[92mTest passed!\033[0m")
