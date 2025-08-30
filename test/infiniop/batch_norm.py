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
    # shape, momentum, eps
    ((13, 4, 5,), 0.1, 1e-5),
    ((2, 3, 4),  0.1, 1e-4),
    ((15, 16, 17,), 0.2, 1e-5),
    ((50, 60, 70),  0.1, 1e-4),
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


# No implement for INPLACE


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


def torch_batch_norm(
    output: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    init_running_mean: torch.Tensor,
    init_running_var: torch.Tensor,
    momentum: float,
    eps: float
):
    bn = torch.nn.BatchNorm1d(
        num_features=input.shape[1],
        eps=eps,
        momentum=momentum,
        dtype=input.dtype,
    )
    bn.weight.data = weight
    bn.bias.data = bias
    bn.running_mean.data = init_running_mean
    bn.running_var.data = init_running_var
    output.copy_(bn(input).detach())
    running_mean.copy_(bn.running_mean.data)
    running_var.copy_(bn.running_var.data)


def test(
    handle,
    device,
    shape, momentum, eps,
    inplace,
    dtype,
    sync=None,
):
    running_mean = TestTensor(
        [shape[1]],
        None,
        dtype,
        device,
    )    
    running_var = TestTensor(
        [shape[1]],
        None,
        dtype,
        device,
    ) 

    input = TestTensor(
        shape,
        None,
        dtype,
        device,
    )   
    if inplace == Inplace.INPLACE:
        output = input
    else:
        output = TestTensor(
            shape,
            None,
            dtype,
            device
        ) 

    weight = TestTensor(
        [shape[1]],
        None,
        dtype,
        device,
    )
    bias = TestTensor(
        [shape[1]],
        None,      
        dtype,
        device,
    )            


    print(
        f"Testing BatchNorm on {InfiniDeviceNames[device]} with shape:{shape}, inplace:{inplace}, momentum:{momentum}, eps:{eps},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    
    torch_batch_norm(
        output.torch_tensor(), running_mean.torch_tensor(), running_var.torch_tensor(),
        input.torch_tensor(), weight.torch_tensor(), bias.torch_tensor(),
        running_mean.torch_tensor(), running_var.torch_tensor(),
        momentum, eps
    )


    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateBatchNormDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            running_mean.descriptor,
            running_var.descriptor,
            input.descriptor,
            weight.descriptor,
            bias.descriptor,
            momentum,
            eps
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [output, running_mean, running_var, input, weight, bias]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetBatchNormWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output.device)

    def lib_batch_norm():
        check_error(
            LIBINFINIOP.infiniopBatchNorm(
                descriptor,
                workspace.data(),
                workspace.size(),
                output.data(),
                running_mean.data(),
                running_var.data(),
                input.data(),
                weight.data(),
                bias.data(),
                None,
            )
        )

    lib_batch_norm()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)
        debug(running_mean.actual_tensor(), running_mean.torch_tensor(), atol=atol, rtol=rtol)
        debug(running_var.actual_tensor(), running_var.torch_tensor(), atol=atol, rtol=rtol)


    assert torch.allclose(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(running_mean.actual_tensor(), running_mean.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(running_var.actual_tensor(), running_var.torch_tensor(), atol=atol, rtol=rtol)
    

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_batch_norm(
            output.torch_tensor(), running_mean.torch_tensor(), running_var.torch_tensor(),
            input.torch_tensor(), weight.torch_tensor(), bias.torch_tensor(), running_mean.torch_tensor(), running_var.torch_tensor(), momentum, eps
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_batch_norm(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyBatchNormDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my BatchNorm passed!\033[0m")
