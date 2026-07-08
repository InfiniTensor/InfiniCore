import ctypes
from ctypes import c_float, c_uint64

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


_CASES = [((4, 512),), ((16, 4096),)]


def test_op(handle, device, shape, dtype=InfiniDtype.BF16, sync=None):
    print(
        f"Testing DSV4 add_rmsnorm_quant on {InfiniDeviceNames[device]} shape:{shape}"
    )
    res = TestTensor(shape, None, dtype, device)
    x = TestTensor(shape, None, dtype, device)
    weight = TestTensor((shape[-1],), None, dtype, device)
    q = TestTensor(shape, None, InfiniDtype.I8, device, mode="zeros")
    rows = res.torch_tensor().numel() // shape[-1]
    scale = TestTensor((rows, 1), None, InfiniDtype.F32, device, mode="zeros")
    eps = 1e-6
    added = res.torch_tensor().float() + x.torch_tensor().float()
    norm = (
        added
        * torch.rsqrt(added.pow(2).mean(-1, keepdim=True) + eps)
        * weight.torch_tensor().float()
    )
    absmax = norm.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    ref_q = torch.round(norm * (127.0 / absmax)).clamp(-128, 127).to(torch.int8)
    ref_s = absmax / 127.0
    ref_res = added.to(res.torch_tensor().dtype)
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4AddRMSNormQuantDescriptor(
            handle,
            ctypes.byref(desc),
            res.descriptor,
            q.descriptor,
            scale.descriptor,
            x.descriptor,
            weight.descriptor,
            c_float(eps),
        )
    )
    for t in [res, x, weight, q, scale]:
        t.destroy_desc()
    ws, wsz = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4AddRMSNormQuantWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4AddRMSNormQuant(
            desc,
            ws.data(),
            wsz,
            res.data(),
            q.data(),
            scale.data(),
            x.data(),
            weight.data(),
            None,
        )
    )
    assert torch.allclose(res.actual_tensor(), ref_res, atol=1e-2, rtol=1e-2)
    assert (q.actual_tensor().float() - ref_q.float()).abs().max().item() <= 1.0
    assert torch.allclose(
        scale.actual_tensor().reshape(ref_s.shape), ref_s, atol=2e-4, rtol=2e-4
    )
    check_error(LIBINFINIOP.infiniopDestroyDsv4AddRMSNormQuantDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 add_rmsnorm_quant Test passed!\033[0m")
