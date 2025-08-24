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
    # shape, grad_weight_strides, grad_bias_strides
    ((5, 4, 3), [2], None), 
    ((6, 10, 5), None, [3]), 
    ((15, 9, 4), [1], [1]), 
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-1, "rtol": 1e-1},
    InfiniDtype.F32: {"atol": 5e-2, "rtol": 5e-2},
    InfiniDtype.BF16: {"atol": 2e-1, "rtol": 2e-1},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_batch_norm_backward(
        grad_input: torch.Tensor,
        grad_weight: torch.Tensor,
        grad_bias: torch.Tensor,
        input, grad_output, weight, running_mean, running_var
    ):
    bn = torch.nn.BatchNorm1d(
        num_features=input.shape[1],
        momentum=1,
        eps=0,
        dtype=torch.float
    ).to(input.device)
    bn.weight.data = weight.type(torch.float)
    bn.running_mean.data = running_mean.type(torch.float)
    bn.running_var.data = running_var.type(torch.float)

    input.requires_grad_(True)
    output = bn(input.type(torch.float))
    output.backward(grad_output)

    grad_input.copy_(input.grad.type(input.dtype))
    grad_weight.copy_(bn.weight.grad.type(input.dtype))
    grad_bias.copy_(bn.bias.grad.type(input.dtype))


def test(
    handle,
    device,
    shape, grad_weight_strides, grad_bias_strides, 
    inplace,
    dtype,
    sync=None,
):
    print(
        f"Testing BatchNorm backward on {InfiniDeviceNames[device]} with shape:{shape},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )    
    grad_weight = TestTensor(
        [shape[1]],
        grad_weight_strides,
        dtype,
        device,
    )
    torch_type = grad_weight._torch_tensor.dtype
    torch_device = grad_weight._torch_tensor.device
    grad_bias = TestTensor(
        [shape[1]],
        grad_bias_strides,
        dtype,
        device,
    )   
    input = TestTensor(
        shape,
        None,
        dtype,
        device,
    )
    torch_input = input.torch_tensor()

    reshape_input = torch_input.permute(1, 0, 2).reshape((shape[1], -1))
    running_mean_torch_tensor = reshape_input.mean(dim=1).to(torch_device).type(torch_type)
    running_var_torch_tensor = reshape_input.var(dim=1, unbiased=False).to(torch_device).type(torch_type)


    grad_output = TestTensor(
        shape,
        None,
        dtype,
        device,
    )          
    if inplace == Inplace.INPLACE:
        grad_input = grad_output
    else:
        grad_input = TestTensor(
            shape,
            None,
            dtype,
            device,
        ) 

 
    weight = TestTensor(
        [shape[1]],
        None,
        dtype,
        device,
    )
    running_mean = TestTensor(
        [shape[1]],
        running_mean_torch_tensor.stride(),
        dtype,
        device,
        "manual",
        set_tensor=running_mean_torch_tensor        
    )
    running_var = TestTensor(
        [shape[1]],
        running_var_torch_tensor.stride(),
        dtype,
        device,
        "manual",
        set_tensor=running_var_torch_tensor        
    )




    
    torch_batch_norm_backward(
        grad_input.torch_tensor(),
        grad_weight.torch_tensor(),
        grad_bias.torch_tensor(),
        input.torch_tensor(),
        grad_output.torch_tensor(),
        weight.torch_tensor(),
        running_mean.torch_tensor(),
        running_var.torch_tensor(),
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateBatchNormBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
            grad_input.descriptor,
            grad_weight.descriptor,
            grad_bias.descriptor,
            input.descriptor,
            grad_output.descriptor,
            weight.descriptor,
            running_mean.descriptor,
            running_var.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input, grad_output, grad_input, grad_weight, grad_bias, weight, running_mean, running_var]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetBatchNormBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, grad_input.device)

    def lib_batch_norm_backward():
        check_error(
            LIBINFINIOP.infiniopBatchNormBackward(
                descriptor,
                workspace.data(),
                workspace.size(),
                grad_input.data(),
                grad_weight.data(),
                grad_bias.data(),
                input.data(),
                grad_output.data(),
                weight.data(),
                running_mean.data(),
                running_var.data(),
                None,
            )
        )

    lib_batch_norm_backward()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(grad_input.actual_tensor(), grad_input.torch_tensor(), atol=atol, rtol=rtol)
        debug(grad_weight.actual_tensor(), grad_weight.torch_tensor(), atol=atol, rtol=rtol)
        debug(grad_bias.actual_tensor(), grad_bias.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(grad_input.actual_tensor(), grad_input.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(grad_weight.actual_tensor(), grad_weight.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(grad_bias.actual_tensor(), grad_bias.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_batch_norm_backward(
            grad_input.torch_tensor(),
            grad_weight.torch_tensor(),
            grad_bias.torch_tensor(),
            input.torch_tensor(),
            grad_output.torch_tensor(),
            weight.torch_tensor(),
            running_mean.torch_tensor(),
            running_var.torch_tensor(),
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_batch_norm_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyBatchNormBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my BatchNorm Backward passed!\033[0m")
