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

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # x_shape, x_stride, x_packed_stride, symmetric
    ((16, 5632), None, None, True),
    ((13, 4), (10, 1), None, True),
    ((13, 4), (10, 1), (10, 1), True),
    ((16, 5632), (13312, 1), (13312, 1), True),
    ((4, 4, 5632), None, None, True),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), True),
    ((1, 4, 132, 128), (67584, 16896, 128, 1), (67584, 16896, 128, 1), True),
    ((1, 4, 132, 128), None, None, True),
]


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 5e-2},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 3e-5, "rtol": 5e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def per_tensor_dequant_int8_torch(x_packed, x_scale, dtype):
    fake_qweight = x_packed.to(dtype)
    dq_weight = fake_qweight * x_scale
    return dq_weight


def test(
    handle,
    device,
    x_shape,
    x_stride,
    x_packed_stride,
    symmetric,
    dtype=InfiniDtype.F16,
    sync=None,
):
    if symmetric == False:
        return
    print(
        f"Testing Per Tensor Dequant Int8 on {InfiniDeviceNames[device]} with x_shape:{x_shape}, x_stride:{x_stride}, x_packed_stride:{x_packed_stride}, symmetric:{symmetric} , dtype:{InfiniDtypeNames[dtype]}"
    )

    x = TestTensor(x_shape, x_stride, dtype, device)

    x_packed = TestTensor(
        x_shape,
        x_packed_stride,
        InfiniDtype.I8,
        device,
        randint_low=-127,
        randint_high=127,
    )
    x_scale = TestTensor((1,), None, InfiniDtype.F32, device)
    if symmetric:
        x_zero = None
    else:
        x_zero = TestTensor((1,), None, InfiniDtype.F32, device)

    ans = per_tensor_dequant_int8_torch(
        x_packed.torch_tensor(), x_scale.torch_tensor(), x.torch_tensor().dtype
    )
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreatePerTensorDequantI8Descriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor,
            x_packed.descriptor,
            x_scale.descriptor,
            None if symmetric else x_zero.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel

    x_packed.destroy_desc()
    x_scale.destroy_desc()
    if symmetric == False:
        x_zero.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetPerTensorDequantI8WorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_per_tensor_dequant_int8():
        check_error(
            LIBINFINIOP.infiniopPerTensorDequantI8(
                descriptor,
                workspace.data(),
                workspace_size.value,
                x.data(),
                x_packed.data(),
                x_scale.data(),
                None if symmetric else x_zero.data(),
                None,
            )
        )

    lib_per_tensor_dequant_int8()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(x.actual_tensor().float(), ans.float(), atol=atol, rtol=rtol)

    assert torch.allclose(x.actual_tensor().float(), ans.float(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: per_tensor_dequant_int8_torch(x_packed.torch_tensor(), x_scale.torch_tensor(), x.torch_tensor().dtype), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_per_tensor_dequant_int8(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyPerTensorDequantI8Descriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
