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


def _workspace(desc, getter, device):
    size = c_size_t(0)
    check_error(getter(desc, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


_LIMIT = 7.0
_CASES = [((7, 512),), ((128, 4096),)]


def _ref(x, limit):
    hidden = x.shape[-1] // 2
    gate = torch.minimum(
        x[..., :hidden], torch.tensor(limit, dtype=x.dtype, device=x.device)
    )
    up = torch.clamp(x[..., hidden:], min=-limit, max=limit)
    return (torch.sigmoid(gate.float()) * gate.float() * up.float()).to(x.dtype)


def test_op(handle, device, shape, dtype=InfiniDtype.BF16, sync=None):
    print(
        f"Testing DSV4 sglang_silu_and_mul_clamp on {InfiniDeviceNames[device]} shape:{shape}"
    )
    torch.manual_seed(shape[0] + shape[1])
    x_torch = torch.randn(shape, dtype=torch.bfloat16, device="cuda").contiguous()
    out_shape = (shape[0], shape[1] // 2)
    x = TestTensor.from_torch(x_torch, dtype, device)
    output = TestTensor(out_shape, None, dtype, device, mode="zeros")
    ref = _ref(x_torch, _LIMIT)
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangSiluAndMulClampDescriptor(
            handle,
            ctypes.byref(desc),
            output.descriptor,
            x.descriptor,
            c_double(_LIMIT),
        )
    )
    for tensor in [x, output]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4SglangSiluAndMulClampWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangSiluAndMulClamp(
            desc, workspace.data(), workspace_size, output.data(), x.data(), None
        )
    )
    assert torch.allclose(output.actual_tensor(), ref, atol=2e-2, rtol=2e-2)
    check_error(LIBINFINIOP.infiniopDestroyDsv4SglangSiluAndMulClampDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 sglang_silu_and_mul_clamp Test passed!\033[0m")
