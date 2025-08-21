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
    (50, 40, True, None, None, [1, 377]),
    (50, 40, False, [10], [1], None),
    (50, 40, True, [10], [1], None),
    (333, 999, True, [1], [10], None),
    (333, 999, False, [1], [10], None),      
    (1001, 505, True, None, None, [3001, 3]),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

    
def torch_linear_backward(
        grad_x: torch.Tensor,
        grad_w: torch.Tensor,
        grad_b: torch.Tensor,
        grad_y: torch.Tensor,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        bias_exist:bool
    ):
    x.requires_grad_(True)
    w.requires_grad_(True)
    if bias_exist:
        b.requires_grad_(True)
    y = torch.nn.functional.linear(x, w, bias=(b if bias_exist else None))
    y.backward(grad_y)
    grad_x.copy_(x.grad)
    grad_w.copy_(w.grad)
    if bias_exist:
        grad_b.copy_(b.grad)

def test(
    handle,
    device,
    in_features, out_features, bias_exist, grad_x_strides, grad_y_strides, grad_w_strides,
    dtype,
    sync=None,
):
    print(
        f"Testing linear_backward on {InfiniDeviceNames[device]} with in_features:{in_features}, out_features: {out_features},"
        f"bias:{bias_exist},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    grad_x = TestTensor(
        [in_features],
        grad_x_strides,
        dtype,
        device,
    )

    grad_w = TestTensor(
        [out_features, in_features],
        grad_w_strides,
        dtype,
        device,
    )

    grad_b = TestTensor(
        [out_features],
        None,
        dtype,
        device,
    ) if bias_exist else None

    grad_y = TestTensor(
        [out_features],
        grad_y_strides,
        dtype,
        device,
    )

    x = TestTensor(
        [in_features],
        None,
        dtype,
        device,
    )

    w = TestTensor(
        [out_features, in_features],
        None,
        dtype,
        device,
    )

    b = TestTensor(
        [out_features],
        None,
        dtype,
        device,
    ) if bias_exist else None




    torch_linear_backward(
        grad_x.torch_tensor(), grad_w.torch_tensor(),
        grad_b.torch_tensor() if bias_exist else None,
        grad_y.torch_tensor(), x.torch_tensor(), w.torch_tensor(),
        b.torch_tensor() if bias_exist else None,
        bias_exist
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateLinearBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
			grad_x.descriptor,
			grad_w.descriptor,
			(grad_b.descriptor if bias_exist else None),
			grad_y.descriptor,
			x.descriptor,
			w.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [grad_x, grad_w, grad_y, x, w,] + [grad_b, b] if bias_exist else []:
        if tensor is not None:
            tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLinearBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, grad_x.device)

    def lib_linear_backward():
        check_error(
            LIBINFINIOP.infiniopLinearBackward(
                descriptor,
                workspace.data(),
                workspace.size(),
				grad_x.data(),
				grad_w.data(),
				grad_b.data() if bias_exist else None,
				grad_y.data(),
				x.data(),
				w.data(),                
                None,
            )
        )

    lib_linear_backward()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(grad_x.actual_tensor(), grad_x.torch_tensor(), atol=atol, rtol=rtol)
        debug(grad_w.actual_tensor(), grad_w.torch_tensor(), atol=atol, rtol=rtol)
        if bias_exist:
            debug(grad_b.actual_tensor(), grad_b.torch_tensor(), atol=atol, rtol=rtol)


    assert torch.allclose(grad_x.actual_tensor(), grad_x.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(grad_w.actual_tensor(), grad_w.torch_tensor(), atol=atol, rtol=rtol)
    if bias_exist:
        assert torch.allclose(grad_b.actual_tensor(), grad_b.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_linear_backward(
            grad_x.torch_tensor(),
            grad_w.torch_tensor(),
            grad_b.torch_tensor() if bias_exist else None,
            grad_y.torch_tensor(),
            x.torch_tensor(),
            w.torch_tensor(),
            b.torch_tensor() if bias_exist else None,
            bias_exist
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_linear_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyLinearBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my linear_backward passed!\033[0m")
