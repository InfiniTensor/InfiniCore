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
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-1, "rtol": 1e-1},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_linear(x, w, b, bias):
    return torch.nn.functional.linear(x, w, bias=(b if bias else None))
    

def test(
    handle,
    device,
    in_features, out_features, bias_exist, x_strides, y_strides, w_strides,
    dtype,
    sync=None,
):
    y = TestTensor(
        [out_features],
        y_strides,
        dtype,
        device,
    )

    x = TestTensor(
        [in_features],
        x_strides,
        dtype,
        device,
    )

    w = TestTensor(
        [out_features, in_features],
        w_strides,
        dtype,
        device,
    )

    b = TestTensor(
        [out_features],
        None,
        dtype,
        device,
    ) if bias_exist else None

    print(
        f"Testing linear on {InfiniDeviceNames[device]} with in_features:{in_features}, out_features:{out_features}, bias:{bias_exist},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    y._torch_tensor = torch_linear(x.torch_tensor(), w.torch_tensor(), (b.torch_tensor() if bias_exist else None), bias_exist)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateLinearDescriptor(
            handle,
            ctypes.byref(descriptor),
			y.descriptor,
			x.descriptor,
			w.descriptor,
			(b.descriptor if bias_exist else None),
        )
    )
    
    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [y, x, w] + ([b] if bias_exist else []):
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLinearWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)
    
    def lib_linear():
        check_error(
            LIBINFINIOP.infiniopLinear(
                descriptor,
                workspace.data(),
                workspace.size(),
				y.data(),
				x.data(),
				w.data(),
				b.data() if bias_exist else None,                
                None,
            )
        )
    
    lib_linear()
    

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)     

    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_linear(
            y.torch_tensor(), x.torch_tensor(), w.torch_tensor(), (b.torch_tensor() if bias_exist else None), bias_exist
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_linear(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyLinearDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my linear passed!\033[0m")
