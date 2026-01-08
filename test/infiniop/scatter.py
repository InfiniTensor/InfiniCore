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
import random

_TEST_CASES = [
    # input_shape, index_shape, output_shape, dim, input_strides, output_strides, index_strides
    ((6, 7), (6, 7), (6, 7), 1, (7, 1), (1, 7), None),
    ((2, 3, 7), (2, 3, 5), (2, 3, 5), 2, (1, 2, 6), None, None),
    ((10, 5, 4), (10, 4, 4), (10, 4, 4), 1, None, None, [16, 4, 1]),
    ((11, 2, 2, 4), (11, 2, 2, 4), (11, 2, 2, 4), 0, None, [16, 8, 4, 1], None),
]


_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.F32: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_scatter(output: torch.Tensor, input, index, dim):
    output.scatter_(dim, index, src=input)
    

def test(
    handle,
    device,
    input_shape, index_shape, output_shape, dim, input_strides, output_strides, index_strides,
    dtype,
    sync=None,
):
    print(
        f"Testing scatter on {InfiniDeviceNames[device]} with input_shape:{input_shape}, index_shape:{index_shape}, output_shape:{output_shape}, dim:{dim},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    output = TestTensor(
        output_shape,
        output_strides,
        dtype,
        device,
        "zeros",
    )

    input = TestTensor(
        input_shape,
        input_strides,
        dtype,
        device,
    )

    def get_test_index_tensor(input_shape, index_shape, output_shape, scatter_dim):
        index = torch.empty(index_shape, dtype=torch.int64)
        ndim = len(input_shape)
        if ndim == 2 and scatter_dim == 1:
            for i in range(input.shape[0]):
                row = list(range(output_shape[dim]))
                random.shuffle(row)
                index[i, :] = torch.tensor(row[:index_shape[dim]]).type(torch.float64)
        elif ndim == 3 and scatter_dim == 2:
            for i in range(input.shape[0]):
                for j in range(input.shape[1]):
                    row = list(range(output_shape[dim]))
                    random.shuffle(row)
                    index[i, j, :] = torch.tensor(row[:index_shape[dim]]).type(torch.float64)
        elif ndim == 3 and scatter_dim == 1:
            for i in range(input.shape[0]):
                for j in range(input.shape[2]):
                    row = list(range(output_shape[dim]))
                    random.shuffle(row)
                    index[i, :, j] = torch.tensor(row[:index_shape[dim]]).type(torch.float64)
        elif ndim == 4 and scatter_dim == 0:
            for i in range(input.shape[1]):
                for j in range(input.shape[2]):
                    for k in range(input.shape[3]):
                        row = list(range(output_shape[dim]))
                        random.shuffle(row)
                        index[:, i, j, k] = torch.tensor(row[:index_shape[dim]]).type(torch.float64)
        return index
    
    torch_index = get_test_index_tensor(input_shape, index_shape, output_shape, dim).type(torch.int64)
    if index_strides:
        torch_index = torch_index.as_strided(index_shape, index_strides)    
    index = TestTensor(
        index_shape,
        torch_index.stride(),
        InfiniDtype.I64,
        device,
        "manual",
        set_tensor=torch_index
    )

    torch_scatter(output.torch_tensor(), input.torch_tensor(), index.torch_tensor(), dim)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateScatterDescriptor(
            handle,
            ctypes.byref(descriptor),
			output.descriptor,
			input.descriptor,
			index.descriptor,
			dim,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [output, input, index]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetScatterWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output.device)

    def lib_scatter():
        check_error(
            LIBINFINIOP.infiniopScatter(
                descriptor,
                workspace.data(),
                workspace.size(),
				output.data(),
				input.data(),
				index.data(),                
                None,
            )
        )

    lib_scatter()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)
    # print('input:\n', input.torch_tensor())
    # print('index:\n', index.torch_tensor())
    # print('output:\n', output.torch_tensor(), '\n', output.actual_tensor(), )


    assert torch.allclose(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_scatter(
            output.torch_tensor(), input.torch_tensor(), index.torch_tensor(), dim
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_scatter(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyScatterDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my scatter passed!\033[0m")
