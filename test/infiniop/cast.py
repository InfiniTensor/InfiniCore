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
_TEST_SHAPES_ = [
    (13, 4),
    (13, 4, 4),
    (16, 5632),
    (4, 4, 5632),
    (1024,),
    (32, 32),
]

_TEST_STRIDES_ = [
    None,  # Contiguous
    # Add some non-contiguous strides for specific shapes
]

# Define type conversion test matrix
_TYPE_CONVERSIONS_ = [
    # Integer to integer conversions
    (InfiniDtype.I32, InfiniDtype.I64),
    (InfiniDtype.I64, InfiniDtype.I32),
    (InfiniDtype.U32, InfiniDtype.U64),
    (InfiniDtype.U64, InfiniDtype.U32),
    (InfiniDtype.I32, InfiniDtype.U32),
    (InfiniDtype.U32, InfiniDtype.I32),
    # Integer to float conversions
    (InfiniDtype.I32, InfiniDtype.F32),
    (InfiniDtype.I32, InfiniDtype.F64),
    (InfiniDtype.I64, InfiniDtype.F32),
    (InfiniDtype.I64, InfiniDtype.F64),
    (InfiniDtype.U32, InfiniDtype.F32),
    (InfiniDtype.U32, InfiniDtype.F64),
    (InfiniDtype.U64, InfiniDtype.F32),
    (InfiniDtype.U64, InfiniDtype.F64),
    # Float to integer conversions
    (InfiniDtype.F32, InfiniDtype.I32),
    (InfiniDtype.F32, InfiniDtype.I64),
    (InfiniDtype.F64, InfiniDtype.I32),
    (InfiniDtype.F64, InfiniDtype.I64),
    (InfiniDtype.F32, InfiniDtype.U32),
    (InfiniDtype.F32, InfiniDtype.U64),
    (InfiniDtype.F64, InfiniDtype.U32),
    (InfiniDtype.F64, InfiniDtype.U64),
    # Float to float conversions
    (InfiniDtype.F32, InfiniDtype.F64),
    (InfiniDtype.F64, InfiniDtype.F32),
    (InfiniDtype.F16, InfiniDtype.F32),
    (InfiniDtype.F32, InfiniDtype.F16),
    (InfiniDtype.F16, InfiniDtype.F64),
    (InfiniDtype.F64, InfiniDtype.F16),
    (InfiniDtype.BF16, InfiniDtype.F32),
    (InfiniDtype.F32, InfiniDtype.BF16),
]

# Form the test cases
_TEST_CASES = []
for input_dtype, output_dtype in _TYPE_CONVERSIONS_:
    for shape in _TEST_SHAPES_:
        for stride in _TEST_STRIDES_:
            _TEST_CASES.append((shape, stride, stride, input_dtype, output_dtype))

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F64: {"atol": 1e-15, "rtol": 1e-15},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.I32: {"atol": 0, "rtol": 0},
    InfiniDtype.I64: {"atol": 0, "rtol": 0},
    InfiniDtype.U32: {"atol": 0, "rtol": 0},
    InfiniDtype.U64: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def cast_pytorch(output, input_tensor):
    """Cast using PyTorch"""
    output.copy_(input_tensor)


def test(
    handle,
    device,
    shape,
    input_stride=None,
    output_stride=None,
    input_dtype=InfiniDtype.F32,
    output_dtype=InfiniDtype.F16,
    sync=None,
):
    # Create input tensor with appropriate data based on type
    if input_dtype in [InfiniDtype.I32, InfiniDtype.I64]:
        # Signed integer: use both positive and negative values
        input_tensor = TestTensor(
            shape, input_stride, input_dtype, device, mode="randint", low=-50, high=50
        )
    elif input_dtype in [InfiniDtype.U32, InfiniDtype.U64]:
        # Unsigned integer: use positive values
        input_tensor = TestTensor(
            shape, input_stride, input_dtype, device, mode="randint", low=0, high=100
        )
    else:
        # Float: use random values
        input_tensor = TestTensor(shape, input_stride, input_dtype, device)

    output_tensor = TestTensor(shape, output_stride, output_dtype, device, mode="zeros")

    print(
        f"Testing Cast on {InfiniDeviceNames[device]} with shape:{shape} "
        f"input_stride:{input_stride} output_stride:{output_stride} "
        f"input_dtype:{InfiniDtypeNames[input_dtype]} output_dtype:{InfiniDtypeNames[output_dtype]}"
    )

    # Perform PyTorch cast for reference
    cast_pytorch(output_tensor.torch_tensor(), input_tensor.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateCastDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            input_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input_tensor, output_tensor]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetCastWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output_tensor.device)

    def lib_cast():
        check_error(
            LIBINFINIOP.infiniopCast(
                descriptor,
                workspace.data(),
                workspace.size(),
                output_tensor.data(),
                input_tensor.data(),
                None,
            )
        )

    lib_cast()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, output_dtype)
    if DEBUG:
        debug(
            output_tensor.actual_tensor(),
            output_tensor.torch_tensor(),
            atol=atol,
            rtol=rtol,
        )

    # For integer types, use exact comparison
    if output_dtype in [
        InfiniDtype.I32,
        InfiniDtype.I64,
        InfiniDtype.U32,
        InfiniDtype.U64,
    ]:
        assert torch.equal(output_tensor.actual_tensor(), output_tensor.torch_tensor())
    else:
        assert torch.allclose(
            output_tensor.actual_tensor(),
            output_tensor.torch_tensor(),
            atol=atol,
            rtol=rtol,
        )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: cast_pytorch(output_tensor.torch_tensor(), input_tensor.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_cast(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyCastDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    print(f"\033[94mRunning Cast operator tests...\033[0m")
    print(f"Total test cases: {len(_TEST_CASES)}")
    print(f"Type conversions tested: {len(_TYPE_CONVERSIONS_)}")
    print("\nType conversion matrix:")
    for i, (input_dtype, output_dtype) in enumerate(_TYPE_CONVERSIONS_):
        print(
            f"  {i+1:2d}. {InfiniDtypeNames[input_dtype]:>6} -> {InfiniDtypeNames[output_dtype]:<6}"
        )
    print()

    for device in get_test_devices(args):
        print(f"\033[93mTesting on device: {InfiniDeviceNames[device]}\033[0m")
        test_operator(
            device, test, _TEST_CASES, []
        )  # Empty dtype list since we handle dtypes in test cases

    print("\033[92mAll Cast tests passed!\033[0m")
