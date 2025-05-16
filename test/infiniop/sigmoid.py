import os

import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64
from libinfiniop import (infiniopHandle_t,
                         infiniopTensorDescriptor_t,
                         open_lib,
                         to_tensor,
                         get_test_devices,
                         check_error,
                         rearrange_if_needed,
                         test_operator,
                         get_args,
                         debug,
                         get_tolerance,
                         profile_operation,
                         create_workspace,
                         )
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, x_stride, y_stride  
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None,),
    ((13, 4, 4), None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), None),
    ((16, 5632), None, None),
    ((16, 5632), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
    ((4, 4, 56320), None, None),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto() 
    INPLACE_X = auto() 


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_X,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
_TENSOR_DTYPES = [torch.float16, torch.float32, torch.float64]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.float32: {"atol": 1e-7, "rtol": 1e-7},
    torch.float64: {"atol": 1e-7, "rtol": 1e-7},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class SigmoidDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopSigmoidDescriptor_t = POINTER(SigmoidDescriptor)


def sigmoid_torch(x):
    return torch.sigmoid(x)


def process_tensors(y, y_strides, x, x_stride, inplace):
    """
    rearrange the tensors if needed and apply the inplace config.
    if inplace is true and the output (i.e., c) is placed to the broadcasted input,
    the inplace config is ignored and out-of-place is used
    """
    original_y_strides = y_strides if y_strides else y.stride()


    def _rearrange(tensor, strides):
        if strides and 0 in strides:
            tensor.set_(tensor.untyped_storage(), 0, tensor.shape, strides)
            return tensor
        else:
            return rearrange_if_needed(tensor, strides)

    x, y = [
        _rearrange(tensor, stride)
        for tensor, stride in zip([x, y], [x_stride, y_strides])
    ]
    y = (
        y
        if inplace == Inplace.OUT_OF_PLACE
        else (x)
    )
    # if inplace is true and c has broadcasted config, reset it to the original unbroadcasted strides
    if 0 in y.stride():
        y.set_(y.untyped_storage(), 0, y.shape, original_y_strides)

    return x, y


def test(lib,
         handle,
         torch_device,
         shape,
         x_stride=None,
         y_stride=None,
         inplace=Inplace.OUT_OF_PLACE,
         dtype=torch.float16,
         sync=None,
         ):
    print(f"Testing Sigmoid on {torch_device} with shape:{shape} x_stride:{x_stride} c_stride:{y_stride} "
          f"dtype:{dtype} inplace:{inplace}")
    '''
    Testing Sigmoid on cpu with shape:(13, 4) a_stride:None b_stride:None c_stride:None 
    dtype:torch.float16 inplace:Inplace.OUT_OF_PLACE
    '''


    x = torch.rand(shape, dtype=dtype).to(torch_device) 
    y = torch.rand(shape, dtype=dtype).to(torch_device)



    x, y = process_tensors(y, y_stride, x, x_stride, inplace)


    ans = sigmoid_torch(x)
    # print("ans",ans)


    x_tensor, = [to_tensor(tensor, lib) for tensor in [x, ]]
    y_tensor = (
        to_tensor(y, lib)
        if inplace == Inplace.OUT_OF_PLACE
        else x_tensor
    )
    if sync is not None:
        sync()

    
    descriptor = infiniopSigmoidDescriptor_t()
    check_error(
        lib.infiniopCreateSigmoidDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x_tensor, y_tensor]:
        tensor.destroyDesc(lib)


    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetSigmoidWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, y.device)

    def lib_sigmoid():
        check_error(
            lib.infiniopSigmoid(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                y_tensor.data,
                x_tensor.data,
                None,
            )
        )

    lib_sigmoid()
    # print("y_tensor", y_tensor.torch_tensor_)
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y, ans, atol=atol, rtol=rtol)

    assert torch.allclose(y, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: sigmoid_torch(x), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_sigmoid(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroySigmoidDescriptor(descriptor))


if __name__ == "__main__":

    args = get_args()
    lib = open_lib()


    lib.infiniopCreateSigmoidDescriptor.restype = c_int32
    lib.infiniopCreateSigmoidDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopSigmoidDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetSigmoidWorkspaceSize.restype = c_int32
    lib.infiniopGetSigmoidWorkspaceSize.argtypes = [
        infiniopSigmoidDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopSigmoid.restype = c_int32
    lib.infiniopSigmoid.argtypes = [
        infiniopSigmoidDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySigmoidDescriptor.restype = c_int32
    lib.infiniopDestroySigmoidDescriptor.argtypes = [
        infiniopSigmoidDescriptor_t,
    ]
 
    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile  # profile是什么意思
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):  # device 是 cpu
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
