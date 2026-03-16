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
    # x_shape, x_stride, x_packed_stride, symmetric, is_static
    ((16, 5632), None, None, True, False),
    ((13, 4), (10, 1), None, True, True),
    ((13, 4), (10, 1), (10, 1), True, False),
    ((16, 5632), (13312, 1), (13312, 1), True, True),
    ((4, 4, 5632), None, None, True, False),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), True, True),
    ((1, 32, 4, 128), (147456, 4608, 128, 1), (147456, 4608, 128, 1), True, False),
    ((1, 32, 4, 128), (16384, 512, 128, 1), (16384, 512, 128, 1), True, True),
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


def per_tensor_quant_int8_torch(x, x_scale, symmetric, is_static):
    if symmetric == False:
        return
    else:
        x = x.float()
        if is_static:
            x_q = x.mul(1 / x_scale)
            x_packed = torch.clamp(x_q, -127, 127).to(torch.int8)
            return x_packed, x_scale, None
        else:
            absmax = x.flatten().abs().max()
            if absmax == 0:
                scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
                q = torch.zeros_like(x, dtype=torch.int8)
                return q, scale, None
        scale = absmax / 127
        x_q = x.mul(127 / absmax)
        x_packed = torch.clamp(x_q, -127, 127).to(torch.int8)
        return x_packed, scale, None


def test(
    handle,
    device,
    x_shape,
    x_stride,
    x_packed_stride,
    symmetric,
    is_static,
    dtype=InfiniDtype.F16,
    sync=None,
):

    print(
        f"Testing Per Tensor Quant Int8 on {InfiniDeviceNames[device]} with x_shape:{x_shape}, x_stride:{x_stride}, x_packed_stride:{x_packed_stride}, symmetric:{symmetric}, is_static:{is_static}, dtype:{InfiniDtypeNames[dtype]}"
    )

    x = TestTensor(x_shape, x_stride, dtype, device)
    x_packed = TestTensor(
        x_shape, x_packed_stride, InfiniDtype.I8, device, mode="zeros"
    )
    if is_static == False:
        x_scale = TestTensor((1,), None, InfiniDtype.F32, device, mode="zeros")
    else:
        x_scale = TestTensor((1,), None, InfiniDtype.F32, device)
    if symmetric:
        x_zero = None
    else:
        x_zero = TestTensor((1,), None, InfiniDtype.F32, device)
    if sync is not None:
        sync()

    x_p, x_s, x_z = per_tensor_quant_int8_torch(
        x.torch_tensor(), x_scale.torch_tensor(), symmetric, is_static
    )

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreatePerTensorQuantI8Descriptor(
            handle,
            ctypes.byref(descriptor),
            x_packed.descriptor,
            x_scale.descriptor,
            None if symmetric else x_zero.descriptor,
            x.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel

    x_packed.destroy_desc()
    x_scale.destroy_desc()
    if symmetric == False:
        x_zero.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetPerTensorQuantI8WorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_per_tensor_quant_int8():
        check_error(
            LIBINFINIOP.infiniopPerTensorQuantI8(
                descriptor,
                workspace.data(),
                workspace_size.value,
                x_packed.data(),
                x_scale.data(),
                None if symmetric else x_zero.data(),
                x.data(),
                is_static,
                None,
            )
        )

    lib_per_tensor_quant_int8()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(x_packed.actual_tensor(), x_p, atol=2, rtol=0)
        debug(x_scale.actual_tensor(), x_s, atol=atol, rtol=rtol)
        if symmetric == False:
            debug(x_zero.actual_tensor(), x_z, atol=atol, rtol=rtol)

    if symmetric:
        assert torch.allclose(
            x_packed.actual_tensor(), x_p, atol=2, rtol=0
        ) and torch.allclose(x_scale.actual_tensor(), x_s, atol=atol, rtol=rtol)
    else:
        assert (
            torch.allclose(x_packed.actual_tensor(), x_p, atol=2, rtol=0)
            and torch.allclose(x_scale.actual_tensor(), x_s, atol=atol, rtol=rtol)
            and torch.allclose(x_zero.actual_tensor(), x_z, atol=atol, rtol=rtol)
        )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: per_tensor_quant_int8_torch(x.torch_tensor(), x_scale.torch_tensor(), symmetric, is_static), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_per_tensor_quant_int8(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyPerTensorQuantI8Descriptor(descriptor))


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
