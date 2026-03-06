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
    # x_shape, symmetric
    ((8, 8), True),
    ((128, 512), True),
    ((128, 128), True),
    ((256, 1024), True),
    ((256, 2048), True),
    ((1024, 2048), True),
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


def per_tensor_quant_int8_torch(x, symmetric):
    if symmetric == False:
        return
    else:
        x = x.float()
        absmax = x.flatten().abs().max()
        if absmax == 0:
            scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
            q = torch.zeros_like(x, dtype=torch.int8)
            return q, scale, None
        scale_x = absmax / 127
        x_q = x.mul(127 / absmax)
        x_q = torch.round(x_q).to(torch.int8)

        return x_q, scale_x, None

def test(
    handle,
    device,
    x_shape,
    symmetric,
    dtype=InfiniDtype.F16,
    sync=None,
):
    
    print(
        f"Testing Per Tensor Quant Int8 on {InfiniDeviceNames[device]} with x_shape:{x_shape}, symmetric:{symmetric} , dtype:{InfiniDtypeNames[dtype]}"
    )
    M, K = x_shape
   
    x = TestTensor(x_shape, None, dtype, device)
    x_p, x_s, x_z = per_tensor_quant_int8_torch(x.torch_tensor(), symmetric)
    x_packed = TestTensor(x_shape, None, InfiniDtype.I8, device, mode="zeros")
    x_scale = TestTensor((1, ), None, InfiniDtype.F32, device)
    if symmetric:
        x_zero = None
    else:
        x_zero = TestTensor((1, ), None, InfiniDtype.F32, device)
    if sync is not None:
        sync()

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
                None,
            )
        )

    lib_per_tensor_quant_int8()
    
    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(x_packed.actual_tensor(), x_p, atol=1, rtol=0)
        debug(x_scale.actual_tensor(), x_s, atol=atol, rtol=rtol)
        if symmetric == False:
            debug(x_zero.actual_tensor(), x_z, atol=atol, rtol=rtol)
    
    if symmetric:
        assert (torch.allclose(x_packed.actual_tensor(), x_p, atol=1, rtol=0) and 
                torch.allclose(x_scale.actual_tensor(), x_s, atol=atol, rtol=rtol))
    else:
        assert (torch.allclose(x_packed.actual_tensor(), x_p, atol=1, rtol=0) and 
                torch.allclose(x_scale.actual_tensor(), x_s, atol=atol, rtol=rtol) and
                torch.allclose(x_zero.actual_tensor(), x_z, atol=atol, rtol=rtol))

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: per_tensor_quant_int8_torch(x.torch_tensor(), symmetric), device, NUM_PRERUN, NUM_ITERATIONS)
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
