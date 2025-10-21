import torch
import ctypes
from ctypes import c_uint64
import torch.nn.functional as F

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
    torch_device_map
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # x_shape, x_stride, topk, norm
    ((1, 10), None, 7, True),
    ((2, 20), None, 4, True),
    ((1, 128), None, 10, True),
]

# w (weight) types
# Note: 'None' means the same as input dtype
_X_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16]  #
# x types used for testing
_VALUE_DTYPES = [InfiniDtype.F32]

# Form the test cases by appending each element of _X_DTYPES to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (x_dtype,) for test_case in _TEST_CASES_ for x_dtype in _X_DTYPES
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def tensorInfo(data):
    print("data:  ", data.is_contiguous(), data.device, data.dtype, data.shape, data.stride(), data.data_ptr(), hex(data.data_ptr()))


def torch_topksoftmax(router_logits, top_k, norm_topk_prob=False):
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights, selected_experts


def test(
        handle,
        device,
        x_shape,
        x_stride,
        topk,
        norm_topk_prob,
        x_dtype=InfiniDtype.F32,
        dtype=InfiniDtype.F16,
        sync=None,
):
    print(
        f"Testing topksoftmax on {InfiniDeviceNames[device]} with x_shape:{x_shape}"
        f"x_stride:{x_stride} w_dtype:{InfiniDtypeNames[x_dtype]} dtype:{InfiniDtypeNames[dtype]}"
    )

    data = torch.arange(0, x_shape[0] * x_shape[1]).reshape(x_shape)

    N, width = x_shape
    x = TestTensor(x_shape, data.stride(), x_dtype, device, scale=0.5, mode="manual", set_tensor=data)

    # print(x.torch_tensor())
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateTopksoftmaxDescriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetTopksoftmaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    values = torch.zeros((N, topk), dtype=torch.float32, device=torch_device_map[x.device])
    indices = torch.zeros((N, topk), dtype=torch.int32, device=torch_device_map[x.device])

    def lib_topksoftmax():
        check_error(
            LIBINFINIOP.infiniopTopksoftmax(
                descriptor,
                workspace.data(),
                workspace_size.value,
                values.data_ptr(),
                indices.data_ptr(),
                x.data(),
                topk,
                norm_topk_prob,
                None,
            )
        )

    lable_values, lable_indices = torch_topksoftmax(x.torch_tensor().clone(), topk, norm_topk_prob=norm_topk_prob)
    lable_indices = lable_indices.to(dtype=torch.int32)
    lib_topksoftmax()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(lable_values, values, atol=atol, rtol=rtol)
        debug(lable_indices, indices, atol=atol, rtol=rtol)

    assert torch.allclose(lable_values, values, atol=atol, rtol=rtol)
    assert torch.allclose(lable_indices, indices, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_topksoftmax(x.actual_tensor().clone(), topk), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_topksoftmax(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyTopksoftmaxDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _VALUE_DTYPES)

    print("\033[92mTest passed!\033[0m")

