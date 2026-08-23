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
# FP8 e4m3fn: symmetric per-tensor quant, scale = max / 448 (dynamic or static)
_TEST_CASES = [
    # x_shape, x_stride, x_packed_stride, is_static, extreme_values
    ((16, 5632), None, None, False, False),
    ((13, 4), (10, 1), None, True, False),
    ((13, 4), (10, 1), (10, 1), False, False),
    ((16, 5632), (13312, 1), (13312, 1), True, False),
    ((4, 4, 5632), None, None, False, False),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), True, False),
    ((1, 32, 4, 128), (147456, 4608, 128, 1), (147456, 4608, 128, 1), False, False),
    ((1, 32, 4, 128), (16384, 512, 128, 1), (16384, 512, 128, 1), True, False),
    # Deterministic boundary coverage: zero, subnormal (< 2^-9), normal,
    # carry rounding, exact max 448, and saturation (|v| > 448 -> 0x7E/0xFE).
    ((1, 16), None, None, True, True),
    ((1, 16), None, None, False, True),
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

FP8_E4M3_MAX = 448.0

# 16 deterministic values covering every e4m3 rounding path:
# 0, subnormal (0.0025 in [2^-9, 2^-8) -> code 0x01), normal, carry rounding,
# exact max 448, and saturation (> 448).
_EXTREME_VALUES = torch.tensor(
    [
        0.0,
        0.0025,
        -0.0025,
        0.5,
        -0.5,
        0.25,
        -0.25,
        2.0,
        3.5,
        100.0,
        -100.0,
        448.0,
        -448.0,
        1000.0,
        -1000.0,
        7.0,
    ],
    dtype=torch.float32,
)


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


def round_half_away(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x + 0.5)


def float_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """float tensor -> e4m3fn byte tensor (mirrors the CUDA kernel)."""
    sign = (x < 0).to(torch.uint8) * 0x80
    ax = x.abs()

    sat = ax >= FP8_E4M3_MAX
    zero = ax < 2.0**-9

    # subnormal: value = m * 2^-9
    sub_m = torch.clamp(round_half_away(ax * 512.0), 0, 7).to(torch.uint8)

    # normal
    e = torch.floor(torch.log2(ax))
    stored = e + 7.0
    frac = ax * torch.exp2(-e) - 1.0
    m = round_half_away(frac * 8.0)
    carry = m == 8.0
    stored2 = torch.where(carry, stored + 1.0, stored)
    m2 = torch.where(carry, torch.zeros_like(m), m)
    sat2 = stored2 > 15.0
    stored3 = torch.clamp(stored2, 0, 15)
    m3 = torch.clamp(m2, 0, 7)
    normal_bits = (stored3.to(torch.uint8) << 3) | m3.to(torch.uint8)

    is_sub = (stored <= 0) & ~zero & ~sat
    bits = torch.where(is_sub, sub_m, normal_bits)
    bits = torch.where(zero, torch.zeros_like(bits), bits)
    bits = torch.where(sat | sat2, torch.full_like(bits, 0x7E), bits)
    return (sign | bits).to(torch.uint8)


def per_tensor_quant_fp8_torch(x, x_scale, is_static):
    x = x.float()
    if is_static:
        scale = x_scale.float()
        x_packed = float_to_e4m3(x / scale).view(torch.float8_e4m3fn)
        return x_packed, scale
    else:
        absmax = x.flatten().abs().max()
        if absmax == 0:
            q = torch.zeros_like(x, dtype=torch.float8_e4m3fn)
            return q, torch.tensor(1.0, device=x.device, dtype=torch.float32)
        scale = absmax / FP8_E4M3_MAX
        x_packed = float_to_e4m3(x / scale).view(torch.float8_e4m3fn)
        return x_packed, scale


def test(
    handle,
    device,
    x_shape,
    x_stride,
    x_packed_stride,
    is_static,
    extreme=False,
    dtype=InfiniDtype.F16,
    sync=None,
):

    print(
        f"Testing Per Tensor Quant Fp8 on {InfiniDeviceNames[device]} with x_shape:{x_shape}, x_stride:{x_stride}, x_packed_stride:{x_packed_stride}, is_static:{is_static}, extreme:{extreme}, dtype:{InfiniDtypeNames[dtype]}"
    )

    if extreme:
        torch_dtype = {
            InfiniDtype.F16: torch.float16,
            InfiniDtype.BF16: torch.bfloat16,
            InfiniDtype.F32: torch.float32,
        }[dtype]
        x = TestTensor(
            x_shape,
            x_stride,
            dtype,
            device,
            mode="manual",
            set_tensor=_EXTREME_VALUES.view(x_shape).to(torch_dtype),
        )
    else:
        x = TestTensor(x_shape, x_stride, dtype, device)
    x_packed = TestTensor(
        x_shape, x_packed_stride, InfiniDtype.F8, device, mode="zeros"
    )
    if extreme and is_static:
        # static scale = 1.0 keeps the raw pattern: |v| / 1.0 = |v|, so values
        # beyond 448 exercise the saturation path.
        x_scale = TestTensor(
            (1,),
            None,
            InfiniDtype.F32,
            device,
            mode="manual",
            set_tensor=torch.tensor([1.0], dtype=torch.float32),
        )
    elif is_static:
        x_scale = TestTensor((1,), None, InfiniDtype.F32, device)
    else:
        x_scale = TestTensor((1,), None, InfiniDtype.F32, device, mode="zeros")
    if sync is not None:
        sync()

    x_p, x_s = per_tensor_quant_fp8_torch(
        x.torch_tensor(), x_scale.torch_tensor(), is_static
    )

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreatePerTensorQuantFp8Descriptor(
            handle,
            ctypes.byref(descriptor),
            x_packed.descriptor,
            x_scale.descriptor,
            x.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_packed.destroy_desc()
    x_scale.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetPerTensorQuantFp8WorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_per_tensor_quant_fp8():
        check_error(
            LIBINFINIOP.infiniopPerTensorQuantFp8(
                descriptor,
                workspace.data(),
                workspace_size.value,
                x_packed.data(),
                x_scale.data(),
                x.data(),
                is_static,
                None,
            )
        )

    lib_per_tensor_quant_fp8()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    # Compare in dequantized space: fp8 rounding-mode differences between the
    # reference and the kernel are within fp8 granularity.
    dq_actual = e4m3_to_float(x_packed.actual_tensor().view(torch.uint8)) * x_scale.actual_tensor()
    dq_ref = e4m3_to_float(x_p.view(torch.uint8)) * x_s
    if DEBUG:
        debug(dq_actual, dq_ref, atol=atol, rtol=rtol)
        debug(x_scale.actual_tensor(), x_s, atol=atol, rtol=rtol)

    # Both sides are fp8 round-trips of the same x; differences are at most one
    # fp8 LSB (rounding-mode), so use an fp8-appropriate tolerance.
    assert torch.allclose(dq_actual, dq_ref, atol=0.02, rtol=0.1) and torch.allclose(
        x_scale.actual_tensor(), x_s, atol=atol, rtol=rtol
    )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: per_tensor_quant_fp8_torch(x.torch_tensor(), x_scale.torch_tensor(), is_static), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_per_tensor_quant_fp8(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyPerTensorQuantFp8Descriptor(descriptor))


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
