import ctypes
from ctypes import c_double, c_size_t

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

_EPS = 1e-6
_CASES = [((1, 1, 512),), ((1, 64, 512),), ((1, 128, 512),)]


def _workspace(desc, getter, device):
    size = c_size_t(0)
    check_error(getter(desc, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


def _ref_rmsnorm(x, eps):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype)


def test_op(handle, device, shape, dtype=InfiniDtype.BF16, sync=None):
    print(f"Testing DSV4 sglang_rmsnorm on {InfiniDeviceNames[device]} shape:{shape}")
    torch.manual_seed(shape[1])
    x_torch = torch.randn(shape, dtype=torch.bfloat16, device="cuda").contiguous()
    x = TestTensor.from_torch(x_torch, dtype, device)
    output = TestTensor(shape, None, dtype, device, mode="zeros")
    ref = _ref_rmsnorm(x_torch, _EPS)

    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangRmsnormDescriptor(
            handle,
            ctypes.byref(desc),
            output.descriptor,
            x.descriptor,
            c_double(_EPS),
        )
    )
    for tensor in [x, output]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc,
        LIBINFINIOP.infiniopGetDsv4SglangRmsnormWorkspaceSize,
        device,
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangRmsnorm(
            desc,
            workspace.data(),
            workspace_size,
            output.data(),
            x.data(),
            None,
        )
    )
    assert torch.allclose(output.actual_tensor(), ref, atol=2e-2, rtol=2e-2)
    check_error(LIBINFINIOP.infiniopDestroyDsv4SglangRmsnormDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 sglang_rmsnorm Test passed!\033[0m")
