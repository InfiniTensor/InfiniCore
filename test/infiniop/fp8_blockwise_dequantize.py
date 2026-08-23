import ctypes
from ctypes import c_uint64

import torch

from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    get_args,
    get_test_devices,
    infiniopOperatorDescriptor_t,
    test_operator,
    to_torch_dtype,
)

_TEST_CASES = [
    # (M, N, BM, BN)
    (256, 256, 128, 128),
    (1024, 1024, 128, 128),
    (384, 1536, 128, 128),
    (512, 768, 64, 128),
    (256, 512, 128, 64),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]


def reference_dequantize(q, scales, block_m, block_n, out_dtype):
    """Torch oracle: decode E4M3 bytes, apply the per-block scale, cast."""
    q_float = q.to(torch.float32)
    scales_full = scales.repeat_interleave(block_m, dim=0).repeat_interleave(block_n, dim=1)
    return (q_float * scales_full).to(out_dtype)


def run_dequantize(handle, device, q_data, scales_data, block_m, block_n, dtype, sync=None):
    q = TestTensor.from_torch(q_data, InfiniDtype.F8, device)
    scales = TestTensor.from_torch(scales_data, InfiniDtype.F32, device)
    out = TestTensor(q_data.shape, None, dtype, device, mode="zeros")
    expected = reference_dequantize(
        q.torch_tensor(), scales.torch_tensor(), block_m, block_n, to_torch_dtype(dtype)
    )

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFp8BlockwiseDequantizeDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            q.descriptor,
            scales.descriptor,
        )
    )
    for tensor in (out, q, scales):
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetFp8BlockwiseDequantizeWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)
    check_error(
        LIBINFINIOP.infiniopFp8BlockwiseDequantize(
            descriptor,
            workspace.data(),
            workspace_size.value,
            out.data(),
            q.data(),
            scales.data(),
            None,
        )
    )
    if sync is not None:
        sync()

    if dtype == InfiniDtype.F16:
        # CPU f32->f16 conversion truncates instead of rounding to nearest, so
        # allow a 1-ulp fp16 gap; BF16 (round-to-nearest-even) and F32 are exact.
        torch.testing.assert_close(out.actual_tensor(), expected, atol=1e-3, rtol=1e-3)
    else:
        torch.testing.assert_close(out.actual_tensor(), expected, atol=0, rtol=0)
    check_error(LIBINFINIOP.infiniopDestroyFp8BlockwiseDequantizeDescriptor(descriptor))


def test(
    handle,
    device,
    m,
    n,
    block_m,
    block_n,
    dtype,
    sync=None,
):
    print(
        f"Testing FP8 blockwise dequantize on {InfiniDeviceNames[device]} "
        f"with q_shape=({m}, {n}), block=({block_m}, {block_n}), "
        f"out_dtype={InfiniDtypeNames[dtype]}"
    )

    probe = TestTensor((1,), None, InfiniDtype.F32, device, mode="zeros")
    torch_device = probe.actual_tensor().device
    # Random floats cover signs, normals, subnormals and zero once quantized to
    # E4M3; all bit patterns produced by .to(torch.float8_e4m3fn) are legal.
    q_data = ((torch.rand(m, n, dtype=torch.float32, device=torch_device) * 2 - 1) * 8).to(
        torch.float8_e4m3fn
    )
    scales_data = (
        torch.rand(m // block_m, n // block_n, dtype=torch.float32, device=torch_device) * 2 + 0.5
    )
    run_dequantize(handle, device, q_data, scales_data, block_m, block_n, dtype, sync)


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(0)
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")
