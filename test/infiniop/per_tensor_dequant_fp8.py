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
    # x_shape, x_stride, x_packed_stride, extreme_bytes
    ((16, 5632), None, None, False),
    ((13, 4), (10, 1), None, False),
    ((13, 4), (10, 1), (10, 1), False),
    ((16, 5632), (13312, 1), (13312, 1), False),
    ((4, 4, 5632), None, None, False),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), False),
    ((1, 4, 132, 128), (67584, 16896, 128, 1), (67584, 16896, 128, 1), False),
    ((1, 4, 132, 128), None, None, False),
    # Deterministic byte coverage: 0, min subnormal, max normal, exact max 448,
    # NaN/inf-reserved patterns (0x7F/0xFF), and both signs of each.
    ((1, 16), None, None, True),
]

_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 5e-2},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 3e-5, "rtol": 5e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def e4m3_to_float(x: torch.Tensor) -> torch.Tensor:
    """e4m3fn byte tensor -> float tensor (mirrors the CUDA kernel)."""
    b = x.view(torch.uint8)
    sign = torch.where((b >> 7) & 1 == 1, -1.0, 1.0)
    exp = (b >> 3) & 0x0F
    mant = b & 0x07
    val = torch.where(
        exp == 0,
        mant.float() * (2.0**-9),
        # exp is uint8: subtract 7 in float to avoid wrapping for exp < 7
        (1.0 + mant.float() / 8.0) * torch.exp2(exp.float() - 7.0),
    )
    return sign * val


# 16 deterministic bytes covering every e4m3 decode path (both signs):
# 0 (0x00/0x80), min/max subnormal (0x01/0x07), min normal (0x08),
# all-mantissa normal (0x77 = 240), exp-15 patterns (0x78 = 256),
# exact max 448 (0x7E/0xFE), NaN/inf-reserved patterns (0x7F/0xFF) which the
# kernel decodes as plain values (480/-480).
_EXTREME_BYTES = torch.tensor(
    [
        0x00,
        0x01,
        0x07,
        0x08,
        0x0F,
        0x10,
        0x3F,
        0x40,
        0x77,
        0x78,
        0x7E,
        0x7F,
        0x80,
        0x81,
        0xFF,
        0xFE,
    ],
    dtype=torch.uint8,
)


def per_tensor_dequant_fp8_torch(x_packed, x_scale, dtype):
    dq = e4m3_to_float(x_packed) * x_scale.float()
    return dq.to(dtype)


def test(
    handle,
    device,
    x_shape,
    x_stride,
    x_packed_stride,
    extreme=False,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing Per Tensor Dequant Fp8 on {InfiniDeviceNames[device]} with x_shape:{x_shape}, x_stride:{x_stride}, x_packed_stride:{x_packed_stride}, extreme:{extreme}, dtype:{InfiniDtypeNames[dtype]}"
    )

    x = TestTensor(x_shape, x_stride, dtype, device, mode="zeros")

    if extreme:
        x_packed = TestTensor(
            x_shape,
            x_packed_stride,
            InfiniDtype.F8,
            device,
            mode="manual",
            set_tensor=_EXTREME_BYTES.view(torch.float8_e4m3fn).view(x_shape),
        )
    else:
        x_packed = TestTensor(
            x_shape,
            x_packed_stride,
            InfiniDtype.F8,
            device,
            mode="float8_e4m3fn",
        )
    x_scale = TestTensor((1,), None, InfiniDtype.F32, device)
    if sync is not None:
        sync()

    ans = per_tensor_dequant_fp8_torch(
        x_packed.torch_tensor(), x_scale.torch_tensor(), x.torch_tensor().dtype
    )

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreatePerTensorDequantFp8Descriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor,
            x_packed.descriptor,
            x_scale.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_packed.destroy_desc()
    x_scale.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetPerTensorDequantFp8WorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_per_tensor_dequant_fp8():
        check_error(
            LIBINFINIOP.infiniopPerTensorDequantFp8(
                descriptor,
                workspace.data(),
                workspace_size.value,
                x.data(),
                x_packed.data(),
                x_scale.data(),
                None,
            )
        )

    lib_per_tensor_dequant_fp8()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(x.actual_tensor().float(), ans.float(), atol=atol, rtol=rtol)

    assert torch.allclose(x.actual_tensor().float(), ans.float(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: per_tensor_dequant_fp8_torch(x_packed.torch_tensor(), x_scale.torch_tensor(), x.torch_tensor().dtype), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_per_tensor_dequant_fp8(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyPerTensorDequantFp8Descriptor(descriptor))


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
