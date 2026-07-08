import ctypes
from ctypes import c_uint64

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

_TEST_CASES = [((1, 4096, 1024),), ((4, 1024, 2048),), ((16, 4096, 1024),)]


def _workspace(descriptor, getter, device):
    size = c_uint64(0)
    check_error(getter(descriptor, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


def test_linear(handle, device, case, dtype=InfiniDtype.BF16, sync=None):
    m, k, n = case
    print(
        f"Testing DSV4 linear_bf16_fp32 on {InfiniDeviceNames[device]} m:{m} k:{k} n:{n}"
    )
    x = TestTensor((m, k), None, dtype, device)
    w = TestTensor((n, k), None, InfiniDtype.F32, device)
    y = TestTensor((m, n), None, InfiniDtype.F32, device, mode="zeros")
    ref = x.torch_tensor().float() @ w.torch_tensor().T
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4LinearBf16Fp32Descriptor(
            handle, ctypes.byref(desc), y.descriptor, x.descriptor, w.descriptor
        )
    )
    for tensor in [x, w, y]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4LinearBf16Fp32WorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4LinearBf16Fp32(
            desc, workspace.data(), workspace_size, y.data(), x.data(), w.data(), None
        )
    )
    assert torch.allclose(y.actual_tensor(), ref, atol=1e-2, rtol=1e-2), (
        (y.actual_tensor() - ref).abs().max().item()
    )
    check_error(LIBINFINIOP.infiniopDestroyDsv4LinearBf16Fp32Descriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_linear, _TEST_CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 linear_bf16_fp32 Test passed!\033[0m")
