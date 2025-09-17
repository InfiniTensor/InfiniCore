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

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # qweight_shape, qzeros_shape, qscales_shape, out_shape, qweight_strides, qzeros_strides,
    # qscales_strides, out_strides, qweights_dtype, qzeros_dtype, qscales_dtype, out_dtype, bits, group_size
    (
        (512, 256),
        (16, 256),
        (16, 2048),
        (512, 2048),
        None,
        None,
        None,
        None,
        InfiniDtype.I32,
        InfiniDtype.I32,
        InfiniDtype.F16,
        InfiniDtype.F16,
        4,
        32,
    ),
    (
        (1024, 128),
        (2, 128),
        (2, 1024),
        (1024, 1024),
        None,
        None,
        None,
        None,
        InfiniDtype.I32,
        InfiniDtype.I32,
        InfiniDtype.F16,
        InfiniDtype.F16,
        4,
        512,
    ),
    (
        (2048, 1024),
        (16, 1024),
        (16, 8192),
        (2048, 8192),
        None,
        None,
        None,
        None,
        InfiniDtype.I32,
        InfiniDtype.I32,
        InfiniDtype.F16,
        InfiniDtype.F16,
        4,
        128,
    ),
    (
        (4096, 512),
        (4, 512),
        (4, 4096),
        (4096, 4096),
        None,
        None,
        None,
        None,
        InfiniDtype.I32,
        InfiniDtype.I32,
        InfiniDtype.F16,
        InfiniDtype.F16,
        4,
        1024,
    ),
    (
        (8192, 256),
        (64, 256),
        (64, 2048),
        (8192, 2048),
        None,
        None,
        None,
        None,
        InfiniDtype.I32,
        InfiniDtype.I32,
        InfiniDtype.F16,
        InfiniDtype.F16,
        4,
        128,
    ),
    (
        (8192, 512),
        (32, 512),
        (32, 4096),
        (8192, 4096),
        None,
        None,
        None,
        None,
        InfiniDtype.I32,
        InfiniDtype.I32,
        InfiniDtype.F16,
        InfiniDtype.F16,
        4,
        256,
    ),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 1e-4},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def dequantize(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    qscales: torch.Tensor,
    bits: int,
    group_size: int,
):
    shifts = torch.arange(0, 32, bits, device=qweight.device)

    # Unpacking qweight columnwise
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    iweights = iweights.view(iweights.shape[0], -1)

    # Unpacking qzeros columnwise
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(
            qzeros[:, :, None], shifts[None, None, :]
        ).to(
            torch.int8  # smallest dtype available
        )
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros

    # Reverse AWQ specific packing order - weights are packed in reverse within each 32-bit word
    reverse_order_tensor = torch.arange(
        iweights.shape[-1],
        dtype=torch.int32,
        device=izeros.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]

    # Extract the actual quantized values by masking higher bits
    iweight = torch.bitwise_and(iweights, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    # Expand scaling factors and zeros to match the full weight dimensions
    # Apply dequantization formula: dequantized = (quantized - zero_point) * scale
    qscales = qscales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)
    iweight = (iweight - izeros) * qscales

    return iweight


# The argument list should be (lib, handle, torch_device, <param list>, dtype)
# The <param list> should keep the same order as the one specified in _TEST_CASES
def test(
    handle,
    device,
    qweights_shape,
    qzeros_shape,
    qscales_shape,
    out_shape,
    qweights_stride,
    qzeros_stride,
    qscales_stride,
    out_stride,
    qweights_dtype,
    qzeros_dtype,
    qscales_dtype,
    out_dtype,
    bits,
    group_size,
    dtype=None,
    sync=None,
):
    print(
        f"Testing Dequantize on {InfiniDeviceNames[device]} with bits:{bits}, group_size:{group_size},"
        f" qweights_shape:{qweights_shape}, qzeros_shape:{qzeros_shape}, qscales_shape:{qscales_shape},"
        f" qweights_stride:{qweights_stride}, qzeros_stride:{qzeros_stride}, qscales_stride:{qscales_stride},"
        f" qweights_dtype:{InfiniDtypeNames[qweights_dtype]}, qzeros_dtype:{InfiniDtypeNames[qzeros_dtype]}, qscales_dtype:{InfiniDtypeNames[qscales_dtype]}"
    )

    qweights = TestTensor(
        qweights_shape, qweights_stride, qweights_dtype, device, mode="randint"
    )
    qzeros = TestTensor(qzeros_shape, qzeros_stride, qzeros_dtype, device, mode="randint")
    qscales = TestTensor(qscales_shape, qscales_stride, qscales_dtype, device)
    out = TestTensor(out_shape, out_stride, out_dtype, device, mode="zeros")
    ans = TestTensor(out_shape, out_stride, out_dtype, device, mode="ones")

    # Compute the PyTorch reference result
    def torch_dequantize():
        return dequantize(
            qweights.torch_tensor(),
            qzeros.torch_tensor(),
            qscales.torch_tensor(),
            bits,
            group_size,
        )

    ans = torch_dequantize()

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDequantizeDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            qweights.descriptor,
            qscales.descriptor,
            qzeros.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [qweights, qzeros, qscales, out]:
        tensor.destroy_desc()

    # Get workspace size and create workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetDequantizeWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Execute infiniop gemm operator
    def lib_dequantize():
        check_error(
            LIBINFINIOP.infiniopDequantize(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                qweights.data(),
                qscales.data(),
                qzeros.data(),
                None,
            )
        )

    lib_dequantize()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_dequantize(), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_dequantize(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyDequantizeDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
