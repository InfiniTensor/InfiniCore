import ctypes
from ctypes import c_int64, c_size_t

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    TestTensor,
    TestWorkspace,
    check_error,
    get_args,
    get_test_devices,
    infiniopOperatorDescriptor_t,
    test_operator,
)

_CASES = [((9, 32, 128, 2),)]


def _workspace(desc, getter, device):
    size = c_size_t(0)
    check_error(getter(desc, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


def test_op(handle, device, shape, dtype=InfiniDtype.BF16, sync=None):
    m, n, k, num_splits = shape
    print(
        "Testing DSV4 deepgemm_tf32_hc_pernorm_gemm on "
        f"{InfiniDeviceNames[device]} shape:{shape}"
    )
    torch.manual_seed(17)
    a_torch = torch.randn((m, k), dtype=torch.bfloat16, device="cuda").contiguous()
    b_torch = torch.randn((n, k), dtype=torch.float32, device="cuda").contiguous()

    a = TestTensor.from_torch(a_torch, InfiniDtype.BF16, device)
    b = TestTensor.from_torch(b_torch, InfiniDtype.F32, device)
    d = TestTensor((num_splits, m, n), None, InfiniDtype.F32, device, mode="zeros")
    sqr_sum = TestTensor((num_splits, m), None, InfiniDtype.F32, device, mode="zeros")

    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4DeepgemmTf32HcPernormGemmDescriptor(
            handle,
            ctypes.byref(desc),
            a.descriptor,
            b.descriptor,
            d.descriptor,
            sqr_sum.descriptor,
            c_int64(num_splits),
        )
    )
    for tensor in [a, b, d, sqr_sum]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc,
        LIBINFINIOP.infiniopGetDsv4DeepgemmTf32HcPernormGemmWorkspaceSize,
        device,
    )
    check_error(
        LIBINFINIOP.infiniopDsv4DeepgemmTf32HcPernormGemm(
            desc,
            workspace.data(),
            workspace_size,
            a.data(),
            b.data(),
            d.data(),
            sqr_sum.data(),
            None,
        )
    )

    expected_d = a_torch.float() @ b_torch.t()
    expected_s = a_torch.float().square().sum(-1)
    actual_d = d.actual_tensor().sum(0)
    actual_s = sqr_sum.actual_tensor().sum(0)
    assert torch.allclose(actual_s, expected_s, rtol=1e-3, atol=1e-3), (
        "DSV4 deepgemm tf32 hc pernorm sqr_sum mismatch: "
        f"maxdiff={(actual_s - expected_s).abs().max().item():.6f}"
    )
    assert torch.allclose(actual_d, expected_d, rtol=1e-2, atol=1e-1), (
        "DSV4 deepgemm tf32 hc pernorm gemm mismatch: "
        f"maxdiff={(actual_d - expected_d).abs().max().item():.6f}"
    )
    check_error(
        LIBINFINIOP.infiniopDestroyDsv4DeepgemmTf32HcPernormGemmDescriptor(desc)
    )


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 deepgemm_tf32_hc_pernorm_gemm Test passed!\033[0m")
