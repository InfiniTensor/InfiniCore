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


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()

_TEST_CASES = [
    # input_shape, output_shape, dim, output_strides, input_strides,
    ([13, 1], [13, 4], 1, [37, 1], [37, 1], Inplace.OUT_OF_PLACE),
    ([1333, 4], [1333, 4], 0, [1, 1333], [1, 2333], Inplace.INPLACE),
    ([1333, 4], [1333, 4], 0, [1, 1333], [1, 2333], Inplace.OUT_OF_PLACE),
    ([133, 23, 53], [133, 23, 53], 1, None, None, Inplace.OUT_OF_PLACE),
    ([133, 23, 13, 53], [133, 23, 13, 53], 2, None, None, Inplace.OUT_OF_PLACE),
]

# Data types used for testing
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


def torch_index_copy_inplace(output, input, index, dim):
    output.index_copy_(dim, index, input.clone())
    

def test(
    handle,
    device,
    input_shape, output_shape, dim, output_strides, input_strides,
    inplace,
    dtype,
    sync=None,
):
    print(
        f"Testing index_copy_inplace on {InfiniDeviceNames[device]} with shape:{input_shape},"
        f"inplace:{inplace},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    input = TestTensor(
        input_shape,
        input_strides,
        dtype,
        device,
    )
    if inplace == Inplace.INPLACE:
        assert output_shape == input_shape
        output = input
    else:
        output = TestTensor(
            output_shape,
            output_strides,
            dtype,
            device,
            "zeros",
        )

    index_list = list(range(output_shape[dim]))
    
    random.shuffle(index_list)
    torch_index = torch.tensor(index_list[:input_shape[dim]], dtype=torch.int64)
    index = TestTensor(
        [input_shape[dim]],
        torch_index.stride(),
        InfiniDtype.I64,
        device,
        "manual",
        set_tensor=torch_index
    )

    torch_index_copy_inplace(output.torch_tensor(), input.torch_tensor(), index.torch_tensor(), dim)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateIndexCopyInplaceDescriptor(
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
        LIBINFINIOP.infiniopGetIndexCopyInplaceWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output.device)

    def lib_index_copy_inplace():
        check_error(
            LIBINFINIOP.infiniopIndexCopyInplace(
                descriptor,
                workspace.data(),
                workspace.size(),
				output.data(),
				input.data(),
				index.data(),                
                None,
            )
        )

    lib_index_copy_inplace()

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
        profile_operation("PyTorch", lambda: torch_index_copy_inplace(
            output.torch_tensor(), input.torch_tensor(), index.torch_tensor(), dim
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_index_copy_inplace(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyIndexCopyInplaceDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my index_copy_inplace passed!\033[0m")
