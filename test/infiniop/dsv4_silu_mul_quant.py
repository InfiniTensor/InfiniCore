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


def _workspace(desc, getter, device):
    size = c_uint64(0)
    check_error(getter(desc, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


_CASES = [((16, 2048),), ((64, 2048),)]


def test_op(handle, device, shape, dtype=InfiniDtype.BF16, sync=None):
    print(f"Testing DSV4 silu_mul_quant on {InfiniDeviceNames[device]} shape:{shape}")
    gate = TestTensor(shape, None, dtype, device)
    up = TestTensor(shape, None, dtype, device)
    q = TestTensor(shape, None, InfiniDtype.I8, device, mode="zeros")
    rows = gate.torch_tensor().numel() // shape[-1]
    scale = TestTensor((rows, 1), None, InfiniDtype.F32, device, mode="zeros")
    h = (
        torch.sigmoid(gate.torch_tensor().float())
        * gate.torch_tensor().float()
        * up.torch_tensor().float()
    )
    absmax = h.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    ref_q = torch.round(h * (127.0 / absmax)).clamp(-128, 127).to(torch.int8)
    ref_s = absmax / 127.0
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SiluMulQuantDescriptor(
            handle,
            ctypes.byref(desc),
            q.descriptor,
            scale.descriptor,
            gate.descriptor,
            up.descriptor,
        )
    )
    for t in [gate, up, q, scale]:
        t.destroy_desc()
    ws, wsz = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4SiluMulQuantWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SiluMulQuant(
            desc, ws.data(), wsz, q.data(), scale.data(), gate.data(), up.data(), None
        )
    )
    assert (q.actual_tensor().float() - ref_q.float()).abs().max().item() <= 1.0
    assert torch.allclose(
        scale.actual_tensor().reshape(ref_s.shape), ref_s, atol=1e-5, rtol=1e-5
    )
    check_error(LIBINFINIOP.infiniopDestroyDsv4SiluMulQuantDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 silu_mul_quant Test passed!\033[0m")
