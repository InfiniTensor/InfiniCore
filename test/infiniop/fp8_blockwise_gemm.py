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
    # (M, N, K, BM, BK)
    (1, 256, 256, 128, 128),
    (2, 256, 512, 128, 128),
    (4, 512, 384, 64, 128),
    (8, 1024, 512, 128, 256),
    (16, 4096, 4096, 128, 128),
    (13, 2048, 1024, 128, 128),
    (32, 768, 1280, 128, 128),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]


def reference_gemm(a, q, scales, block_n, block_k):
    """Torch oracle in fp32: dequantize per block, then matmul."""
    w = q.to(torch.float32) * scales.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
    return a.to(torch.float32) @ w.t()


def run_case(handle, device, a_data, q_data, scales_data, block_n, block_k, dtype, sync=None):
    a = TestTensor.from_torch(a_data, dtype, device)
    q = TestTensor.from_torch(q_data, InfiniDtype.F8, device)
    scales = TestTensor.from_torch(scales_data, InfiniDtype.F32, device)
    out = TestTensor((a_data.shape[0], q_data.shape[0]), None, dtype, device, mode="zeros")
    expected = reference_gemm(a.torch_tensor(), q.torch_tensor(), scales.torch_tensor(), block_n, block_k)

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFp8BlockwiseGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            a.descriptor,
            q.descriptor,
            scales.descriptor,
        )
    )
    for tensor in (out, a, q, scales):
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetFp8BlockwiseGemmWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = TestWorkspace(workspace_size.value, device)
    check_error(
        LIBINFINIOP.infiniopFp8BlockwiseGemm(
            descriptor,
            workspace.data(),
            workspace_size.value,
            out.data(),
            a.data(),
            q.data(),
            scales.data(),
            None,
        )
    )
    if sync is not None:
        sync()

    actual = out.actual_tensor().to(torch.float32)
    rel = (actual - expected).abs().mean() / expected.abs().mean().clamp(min=1e-6)
    limit = 1e-3 if dtype == InfiniDtype.F32 else 2e-2
    assert rel < limit, f"relative error {rel.item():.6f} >= {limit}"
    check_error(LIBINFINIOP.infiniopDestroyFp8BlockwiseGemmDescriptor(descriptor))


def test(handle, device, m, n, k, block_n, block_k, dtype, sync=None):
    print(
        f"Testing FP8 blockwise GEMM on {InfiniDeviceNames[device]} "
        f"with M={m}, N={n}, K={k}, block=({block_n}, {block_k}), "
        f"dtype={InfiniDtypeNames[dtype]}"
    )
    torch_dtype = to_torch_dtype(dtype)
    probe = TestTensor((1,), None, InfiniDtype.F32, device, mode="zeros")
    torch_device = probe.actual_tensor().device

    a_data = torch.randn(m, k, dtype=torch_dtype, device=torch_device) * 0.5
    q_data = ((torch.rand(n, k, dtype=torch.float32, device=torch_device) * 2 - 1) * 4).to(
        torch.float8_e4m3fn
    )
    scales_data = (
        torch.rand(n // block_n, k // block_k, dtype=torch.float32, device=torch_device) * 0.02 + 0.005
    )
    run_case(handle, device, a_data, q_data, scales_data, block_n, block_k, dtype, sync)


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(0)
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")
