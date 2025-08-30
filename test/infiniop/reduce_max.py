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

_TEST_CASES = [
    # input_shape, output_shape, dim, input_strides, output_strides
    ((13, 4), (13, 1), 1, (4, 1), (1, 1)),
    ((13, 4), (1, 4), 0, (10, 1), (10, 1)),
    ((13, 4, 4), (13, 4, 1), 2, None, None),
    ((16, 5632), (16, 1), 1, None, None),
    ((4, 4, 5632), (1, 4, 5632), 0, None, None),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_reduce_max(output, input, dim):
    return torch.max(input, dim, keepdim=True,)[0]


def test(
    handle,
    device,     
    input_shape, output_shape, dim, input_strides, output_strides,
    dtype=InfiniDtype.F32,
    sync=None,
):
    print(
        f"Testing reduce_max on {InfiniDeviceNames[device]} with input_shape:{input_shape}, dim:{dim},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )
    output = TestTensor(
        output_shape,
        output_strides,
        dtype,
        device,
    )

    input = TestTensor(
        input_shape,
        input_strides,
        dtype,
        device,
    )

    output._torch_tensor = torch_reduce_max(output.torch_tensor(), input.torch_tensor(), dim)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateReduceMaxDescriptor(
            handle,
            ctypes.byref(descriptor),
			output.descriptor,
			input.descriptor,
            dim,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input, output]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetReduceMaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output.device)

    def lib_reduce_max():
        check_error(
            LIBINFINIOP.infiniopReduceMax(
                descriptor,
                workspace.data(),
                workspace.size(),
                output.data(),
                input.data(),
                None,
            )
        )

    lib_reduce_max()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_reduce_max(
            output.torch_tensor(), input.torch_tensor(), dim
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_reduce_max(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyReduceMaxDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my ReduceMax passed!\033[0m")
