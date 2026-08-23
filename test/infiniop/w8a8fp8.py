import torch
import ctypes
from ctypes import c_uint64
from enum import Enum, auto
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
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
# W8A8-FP8: activations are quantized to FP8 e4m3 (per-tensor, dynamic
# scale = max/448). Weights are either kept in fp16 (activation-only FP8,
# the KV-cache-style use case) or quantized to FP8 as well (full symmetric
# W8A8-FP8); everything is dequantized and multiplied with an fp16 matmul.


class WMode(Enum):
    FP16 = auto()  # only activations are fp8-quantized; weights stay fp16
    FP8 = auto()   # both activations and weights are fp8-quantized


_TEST_CASES_ = [
    # x_shape = [M,K], w_shape = [N, K], y_shape = [M, N]
    ((100, 3584), (10752, 3584), (100, 10752)),
    ((1000, 3584), (10752, 3584), (1000, 10752)),
    ((1, 3584), (10752, 3584), (1, 10752)),
    ((2000, 3584), (10752, 3584), (2000, 10752)),
]

_WMODES = [
    WMode.FP16,
    WMode.FP8,
]

_TEST_CASES = [
    test_case + (wmode,)
    for test_case in _TEST_CASES_
    for wmode in _WMODES
]

_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16]

# fp8 e4m3 has ~6% relative precision; allow round-trip + rounding-mode slack.
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 3e-1, "rtol": 1.5e-1},
    InfiniDtype.BF16: {"atol": 3e-1, "rtol": 1.5e-1},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

FP8_E4M3_MAX = 448.0


def e4m3_to_float(x: torch.Tensor) -> torch.Tensor:
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


def per_tensor_quant_fp8_torch(x):
    """Dynamic per-tensor FP8 quant: returns (packed fp8, scale = max/448)."""
    x = x.float()
    absmax = x.flatten().abs().max()
    if absmax == 0:
        scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        q = torch.zeros_like(x, dtype=torch.float8_e4m3fn)
        return q, scale
    scale = absmax / FP8_E4M3_MAX
    # round half away from zero, then convert via fp16 (values fit fp8 range)
    x_q = torch.floor((x / scale).abs() + 0.5) * torch.sign(x)
    q = torch.clamp(x_q, -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def test(
    handle,
    device,
    x_shape,
    w_shape,
    y_shape,
    wmode=WMode.FP8,
    dtype=InfiniDtype.BF16,
    sync=None,
):
    print(
        f"Testing W8A8-Fp8 ({wmode.name}) on {InfiniDeviceNames[device]} with x_shape:{x_shape}, w_shape:{w_shape}, dtype:{InfiniDtypeNames[dtype]}"
    )
    M, K = x_shape
    N = w_shape[0]
    out_dtype = torch.float16 if dtype == InfiniDtype.F16 else torch.bfloat16

    x = TestTensor(x_shape, None, dtype, device)
    w = TestTensor(w_shape, None, dtype, device)

    x_packed = TestTensor(x_shape, None, InfiniDtype.F8, device, mode="zeros")
    x_scale = TestTensor((1,), None, InfiniDtype.F32, device, mode="zeros")
    y = TestTensor(y_shape, None, dtype, device, mode="zeros")

    if wmode == WMode.FP8:
        w_packed = TestTensor(w_shape, None, InfiniDtype.F8, device, mode="zeros")
        w_scale = TestTensor((1,), None, InfiniDtype.F32, device, mode="zeros")

    # Reference: x is always quantized (dynamic per-tensor fp8); w is quantized
    # only in full W8A8 mode, otherwise kept in fp16.
    x_p, x_s = per_tensor_quant_fp8_torch(x.torch_tensor())
    if wmode == WMode.FP8:
        w_p, w_s = per_tensor_quant_fp8_torch(w.torch_tensor())
        w_ref = e4m3_to_float(w_p) * w_s
    else:
        w_ref = w.torch_tensor().float()
    ref = torch.matmul(e4m3_to_float(x_p) * x_s, w_ref.t()).to(out_dtype)

    # --- per_tensor_quant_fp8 on x ---
    quant_x_desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreatePerTensorQuantFp8Descriptor(
            handle,
            ctypes.byref(quant_x_desc),
            x_packed.descriptor,
            x_scale.descriptor,
            x.descriptor,
        )
    )

    # --- per_tensor_dequant_fp8 for x ---
    dequant_x_desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreatePerTensorDequantFp8Descriptor(
            handle,
            ctypes.byref(dequant_x_desc),
            x.descriptor,  # output buffer (fp16/bf16)
            x_packed.descriptor,
            x_scale.descriptor,
        )
    )

    # Invalidate the tensor descriptors now that both ops are created
    x_packed.destroy_desc()
    x_scale.destroy_desc()

    quant_x_ws = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetPerTensorQuantFp8WorkspaceSize(
            quant_x_desc, ctypes.byref(quant_x_ws)
        )
    )
    quant_x_ws_t = TestWorkspace(quant_x_ws.value, x.device)

    def lib_quant_x():
        check_error(
            LIBINFINIOP.infiniopPerTensorQuantFp8(
                quant_x_desc,
                quant_x_ws_t.data(),
                quant_x_ws.value,
                x_packed.data(),
                x_scale.data(),
                x.data(),
                False,  # is_static=False -> dynamic
                None,
            )
        )

    dequant_x_ws = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetPerTensorDequantFp8WorkspaceSize(
            dequant_x_desc, ctypes.byref(dequant_x_ws)
        )
    )
    dequant_x_ws_t = TestWorkspace(dequant_x_ws.value, x.device)

    def lib_dequant_x():
        check_error(
            LIBINFINIOP.infiniopPerTensorDequantFp8(
                dequant_x_desc,
                dequant_x_ws_t.data(),
                dequant_x_ws.value,
                x.data(),  # NOTE: x is both input (pre-quant) and dequant output buffer
                x_packed.data(),
                x_scale.data(),
                None,
            )
        )

    if wmode == WMode.FP8:
        # --- per_tensor_quant_fp8 on w ---
        quant_w_desc = infiniopOperatorDescriptor_t()
        check_error(
            LIBINFINIOP.infiniopCreatePerTensorQuantFp8Descriptor(
                handle,
                ctypes.byref(quant_w_desc),
                w_packed.descriptor,
                w_scale.descriptor,
                w.descriptor,
            )
        )

        # --- per_tensor_dequant_fp8 for w ---
        dequant_w_desc = infiniopOperatorDescriptor_t()
        check_error(
            LIBINFINIOP.infiniopCreatePerTensorDequantFp8Descriptor(
                handle,
                ctypes.byref(dequant_w_desc),
                w.descriptor,
                w_packed.descriptor,
                w_scale.descriptor,
            )
        )

        # Invalidate the tensor descriptors now that both ops are created
        w_packed.destroy_desc()
        w_scale.destroy_desc()

        quant_w_ws = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetPerTensorQuantFp8WorkspaceSize(
                quant_w_desc, ctypes.byref(quant_w_ws)
            )
        )
        quant_w_ws_t = TestWorkspace(quant_w_ws.value, w.device)

        def lib_quant_w():
            check_error(
                LIBINFINIOP.infiniopPerTensorQuantFp8(
                    quant_w_desc,
                    quant_w_ws_t.data(),
                    quant_w_ws.value,
                    w_packed.data(),
                    w_scale.data(),
                    w.data(),
                    False,
                    None,
                )
            )

        dequant_w_ws = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetPerTensorDequantFp8WorkspaceSize(
                dequant_w_desc, ctypes.byref(dequant_w_ws)
            )
        )
        dequant_w_ws_t = TestWorkspace(dequant_w_ws.value, w.device)

        def lib_dequant_w():
            check_error(
                LIBINFINIOP.infiniopPerTensorDequantFp8(
                    dequant_w_desc,
                    dequant_w_ws_t.data(),
                    dequant_w_ws.value,
                    w.data(),
                    w_packed.data(),
                    w_scale.data(),
                    None,
                )
            )

    def lib_w8a8fp8():
        # quant x (and, in full W8A8 mode, w) to dynamic per-tensor fp8,
        # dequant them back into the original buffers, then matmul on the
        # dequantized values.
        lib_quant_x()
        lib_dequant_x()
        if wmode == WMode.FP8:
            lib_quant_w()
            lib_dequant_w()

    lib_w8a8fp8()

    if sync is not None:
        sync()

    # Final matmul on the dequantized values (fp16/bf16)
    y_actual = torch.matmul(
        x.torch_tensor().float(), w.torch_tensor().float().t()
    ).to(out_dtype)

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y_actual, ref, atol=atol, rtol=rtol)

    assert torch.allclose(y_actual.float(), ref.float(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        def profile_operation(name, func, device, num_prerun, num_iterations):
            for _ in range(num_prerun):
                func()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_iterations):
                func()
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            print(f"{name} took {elapsed / num_iterations:.6f} ms over {num_iterations} iterations")

        profile_operation("lib w8a8fp8", lambda: lib_w8a8fp8(), device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyPerTensorQuantFp8Descriptor(quant_x_desc))
    check_error(LIBINFINIOP.infiniopDestroyPerTensorDequantFp8Descriptor(dequant_x_desc))
    if wmode == WMode.FP8:
        check_error(LIBINFINIOP.infiniopDestroyPerTensorQuantFp8Descriptor(quant_w_desc))
        check_error(LIBINFINIOP.infiniopDestroyPerTensorDequantFp8Descriptor(dequant_w_desc))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
