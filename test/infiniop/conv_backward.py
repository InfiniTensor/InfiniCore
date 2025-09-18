import torch
import ctypes
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
from typing import List, Tuple
import math

_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16]
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-4},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}
DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

_TEST_CASES = [
    # 1D Conv Backward Tests
    # x_shape, x_stride, w_shape, w_stride, pads, strides, dilations, group
    ((2, 4, 16), (64, 16, 1), (8, 4, 5), (20, 5, 1), (0,), (1,), (1,), 1),
    ((2, 4, 32), (128, 32, 1), (8, 4, 3), (12, 3, 1), (1,), (2,), (1,), 1),
    ((1, 2, 64), (128, 64, 1), (4, 2, 7), (14, 7, 1), (2,), (3,), (1,), 1),
    # 2D Conv Backward Tests
    (
        (2, 3, 10, 10),
        (300, 100, 10, 1),
        (6, 3, 3, 3),
        (27, 9, 3, 1),
        (0, 1),
        (1, 2),
        (2, 2),
        1,
    ),
    (
        (1, 2, 28, 14),
        (784, 392, 14, 1),
        (4, 2, 5, 3),
        (30, 15, 3, 1),
        (1, 2),
        (3, 2),
        (1, 1),
        1,
    ),
    # 3D Conv Backward Tests
    (
        (1, 2, 5, 5, 5),
        (250, 125, 25, 5, 1),
        (4, 2, 2, 2, 2), 
        (16, 8, 4, 2, 1),
        (0, 1, 1), 
        (1, 1, 1),             
        (1, 1, 1),                   
        1,
    ),
    # Grouped convolution test case
    ((2, 4, 16), (64, 16, 1), (4, 2, 3), (6, 3, 1), (1,), (1,), (1,), 2),
]


def inferShapeStride(
    x_shape: List[int],
    w_shape: List[int],
    pads: List[int],
    strides: List[int],
    dilations: List[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    assert (
        len(x_shape)
        == len(w_shape)
        == len(pads) + 2
        == len(dilations) + 2
        == len(strides) + 2
    ), "x and w should have the same length; pads, strides, and dilatinos should have the same length; the length of pads should be that of x - 2"
    output_dims = [
        math.floor(
            (x_shape[i + 2] + 2 * pads[i] - dilations[i] * (w_shape[i + 2] - 1) - 1)
            / strides[i]
            + 1
        )
        for i in range(len(pads))
    ]
    output_shape = (x_shape[0], w_shape[0]) + tuple(output_dims)
    output_strides = [1]
    for s in reversed(output_shape[1:]):
        output_strides.insert(0, output_strides[0] * s)
    output_strides = tuple(output_strides)
    return output_shape, output_strides


def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)


def test(
    handle,
    device,
    input_shape,
    input_stride,
    weight_shape,
    weight_stride,
    pads,
    strides,
    dilations,
    groups,
    tensor_dtype=InfiniDtype.F16,
    sync=None,
):
    assert len(pads) == len(strides) == len(dilations)
    input = TestTensor(
        input_shape, input_stride, dt=tensor_dtype, device=device, scale=0.01
    )
    weight = TestTensor(
        weight_shape, weight_stride, dt=tensor_dtype, device=device, scale=0.01
    )
    output_shape, output_stride = inferShapeStride(
        input_shape, weight_shape, pads, strides, dilations
    )
    # grad_output = TestTensor(output_shape, output_stride, dt=tensor_dtype, device=device)
    bias = TestTensor(
        (weight.shape[0],), (1,), dt=tensor_dtype, device=device, scale=0.01
    )
    # bias = None  # Disable bias for now
    # 1. PyTorch reference backward
    input_torch = input.torch_tensor().detach().clone().requires_grad_(True)
    weight_torch = weight.torch_tensor().detach().clone().requires_grad_(True)
    bias_torch = (
        bias.torch_tensor().detach().clone().requires_grad_(True)
        if bias is not None
        else None
    )
    grad_output_torch = torch.randn(
        output_shape, dtype=input_torch.dtype, device=input_torch.device
    )

    # Define forward function for reuse
    def forward_pass(input_t, weight_t, bias_t):
        if len(input_shape) == 3:
            return torch.nn.functional.conv1d(
                input_t,
                weight_t,
                bias=bias_t,
                stride=strides,
                padding=pads,
                dilation=dilations,
                groups=groups,
            )
        elif len(input_shape) == 4:
            return torch.nn.functional.conv2d(
                input_t,
                weight_t,
                bias=bias_t,
                stride=strides,
                padding=pads,
                dilation=dilations,
                groups=groups,
            )
        elif len(input_shape) == 5:
            return torch.nn.functional.conv3d(
                input_t,
                weight_t,
                bias=bias_t,
                stride=strides,
                padding=pads,
                dilation=dilations,
                groups=groups,
            )
        else:
            raise NotImplementedError("Unsupported ndim")

    # Forward
    y_ref = forward_pass(input_torch, weight_torch, bias_torch)
    print(
        f"PyTorch output shape: {y_ref.shape}, dtype: {y_ref.dtype}, device: {y_ref.device}"
    )
    y_ref.backward(grad_output_torch)
    grad_input_ref = input_torch.grad
    grad_weight_ref = weight_torch.grad
    grad_bias_ref = bias_torch.grad if bias is not None else None

    # 2. infiniop backward
    grad_output_tensor = TestTensor(
        output_shape, output_stride, dt=tensor_dtype, device=device
    )
    grad_output_tensor.actual_tensor().copy_(grad_output_torch)
    grad_input = TestTensor(input_shape, input_stride, dt=tensor_dtype, device=device)
    grad_weight = TestTensor(
        weight_shape, weight_stride, dt=tensor_dtype, device=device
    )
    grad_bias = (
        TestTensor((weight.shape[0],), (1,), dt=tensor_dtype, device=device)
        if bias is not None
        else None
    )

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateConvBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
            grad_output_tensor.descriptor,
            input.descriptor,
            weight.descriptor,
            bias.descriptor if bias is not None else None,
            tuple_to_void_p(pads),
            tuple_to_void_p(strides),
            tuple_to_void_p(dilations),
            groups,
        )
    )

    for tensor in [
        input,
        grad_output_tensor,
        weight,
        bias,
        grad_input,
        grad_weight,
        grad_bias,
    ]:
        if tensor is not None:
            tensor.destroy_desc()

    workspace_size = ctypes.c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetConvBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, input.device)

    def lib_conv_backward():
        check_error(
            LIBINFINIOP.infiniopConvBackward(
                descriptor,
                workspace.data(),
                workspace_size.value,
                grad_input.data(),
                grad_weight.data(),
                grad_bias.data() if grad_bias is not None else None,
                grad_output_tensor.data(),
                input.data(),
                weight.data(),
                None,
            )
        )

    lib_conv_backward()
    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    # Compare grad_input
    if DEBUG:
        debug(grad_input.actual_tensor(), grad_input_ref, atol=atol, rtol=rtol)
    assert torch.allclose(
        grad_input.actual_tensor(), grad_input_ref, atol=atol, rtol=rtol
    )
    # Compare grad_weight
    if DEBUG:
        debug(grad_weight.actual_tensor(), grad_weight_ref, atol=atol, rtol=rtol)
    assert torch.allclose(
        grad_weight.actual_tensor(), grad_weight_ref, atol=atol, rtol=rtol
    )
    # Compare grad_bias
    if grad_bias is not None:
        if DEBUG:
            debug(grad_bias.actual_tensor(), grad_bias_ref, atol=atol, rtol=rtol)
        assert torch.allclose(
            grad_bias.actual_tensor(), grad_bias_ref, atol=atol, rtol=rtol
        )

    if PROFILE:
        # PyTorch backward function that recreates the computation graph each time
        def torch_conv_backward():
            # Recreate tensors with gradients for each call
            input_t = input.torch_tensor().detach().clone().requires_grad_(True)
            weight_t = weight.torch_tensor().detach().clone().requires_grad_(True)
            bias_t = (
                bias.torch_tensor().detach().clone().requires_grad_(True)
                if bias is not None
                else None
            )
            # Forward pass
            y = forward_pass(input_t, weight_t, bias_t)
            # Backward pass
            y.backward(grad_output_torch)

        # fmt: off
        profile_operation("PyTorch", torch_conv_backward, device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_conv_backward, device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyConvBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mConvBackward test passed!\033[0m")
