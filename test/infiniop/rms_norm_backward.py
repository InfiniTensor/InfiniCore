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

_TEST_CASES_ = [
    # shape, grad_x_strides, grad_w_strides, grad_y_strides, x_strides
    ([13, 4, 5], [37, 5, 1], [2], [38, 5, 1], [39, 5, 1]),
    ([20, 30, 40], [1555, 40, 1], None, None, None),
    ([55, 65, 10], None, [10], None, None),
    ([155, 165, 110], None, None, [40037, 110, 1], None),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]


# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_rms_norm_backward(
    grad_x: torch.Tensor,
    grad_w: torch.Tensor,
    grad_y, x, w
):
    rmsNorm = torch.nn.RMSNorm(
        normalized_shape=[x.shape[-1]],
        eps=0,
        dtype=torch.float
    )
    x = x.type(torch.float)
    x.requires_grad_(True)
    rmsNorm.weight.data = w.type(torch.float)
    y = rmsNorm(x)
    y.backward(grad_y)
    grad_x.copy_(x.grad.type(grad_x.dtype))
    grad_w.copy_(rmsNorm.weight.grad.type(grad_w.dtype))
    

def test(
    handle,
    device,
    input_shape, grad_x_strides, grad_w_strides, grad_y_strides, x_strides,
    inplace,
    dtype,
    sync=None,
):
    print(
        f"Testing rms_norm_backward on {InfiniDeviceNames[device]} with input_shape: {input_shape},"
        f"inplace:{inplace},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    grad_w = TestTensor(
        [input_shape[-1]],
        grad_w_strides,
        dtype,
        device,
    )
    grad_y = TestTensor(
        input_shape,
        grad_y_strides,
        dtype,
        device,
    )
    x = TestTensor(
        input_shape,
        x_strides,
        dtype,
        device,
    )
    if inplace == Inplace.INPLACE:
        if grad_x_strides != grad_y_strides:
            return
        grad_x = grad_y
    else:
        grad_x = TestTensor(
            input_shape,
            grad_x_strides,
            dtype,
            device,
        )
    w = TestTensor(
        [input_shape[-1]],
        None,
        dtype,
        device,
    )

    torch_rms_norm_backward(grad_x.torch_tensor(), grad_w.torch_tensor(), grad_y.torch_tensor(), x.torch_tensor(), w.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRMSNormBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
			grad_x.descriptor,
			grad_w.descriptor,
			grad_y.descriptor,
			x.descriptor,
			w.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [grad_x, grad_w, grad_y, x, w]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRMSNormBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, grad_x.device)

    def lib_rms_norm_backward():
        check_error(
            LIBINFINIOP.infiniopRMSNormBackward(
                descriptor,
                workspace.data(),
                workspace.size(),
				grad_x.data(),
				grad_w.data(),
				grad_y.data(),
				x.data(),
				w.data(),                
                None,
            )
        )

    lib_rms_norm_backward()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(grad_x.actual_tensor(), grad_x.torch_tensor(), atol=atol, rtol=rtol)
        debug(grad_w.actual_tensor(), grad_w.torch_tensor(), atol=atol, rtol=rtol)
    # print('grad_y:\n', grad_y.torch_tensor())
    # print('x:\n', x.torch_tensor())
    # print('w:\n', w.torch_tensor())
    # print('grad_x:\n', grad_x.torch_tensor(), '\n', grad_x.actual_tensor(), )
    # print('grad_w:\n', grad_w.torch_tensor(), '\n', grad_w.actual_tensor(), )


    assert torch.allclose(grad_x.actual_tensor(), grad_x.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(grad_w.actual_tensor(), grad_w.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_rms_norm_backward(
            grad_x.torch_tensor(), grad_w.torch_tensor(), grad_y.torch_tensor(), x.torch_tensor(), w.torch_tensor()
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_rms_norm_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyRMSNormBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my rms_norm_backward passed!\033[0m")
