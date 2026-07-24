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
    print(f"Testing DSV4 act_quant_fp8 on {InfiniDeviceNames[device]} shape:{shape}")
    x = TestTensor(shape, None, dtype, device)
    xq = TestTensor(shape, None, InfiniDtype.F8, device, mode="zeros")
    rows = x.torch_tensor().numel() // shape[-1]
    scale = TestTensor((rows, 1), None, InfiniDtype.F32, device, mode="zeros")
    fp8_max = 448.0
    absmax = x.torch_tensor().float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    ref_s = absmax / fp8_max
    ref_q = (x.torch_tensor().float() / ref_s).to(torch.float8_e4m3fn)
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4ActQuantFp8Descriptor(
            handle,
            ctypes.byref(desc),
            xq.descriptor,
            scale.descriptor,
            x.descriptor,
            c_float(fp8_max),
        )
    )
    for t in [x, xq, scale]:
        t.destroy_desc()
    ws, wsz = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4ActQuantFp8WorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4ActQuantFp8(
            desc, ws.data(), wsz, xq.data(), scale.data(), x.data(), None
        )
    )
    assert torch.allclose(
        scale.actual_tensor().reshape(ref_s.shape), ref_s, atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(xq.actual_tensor().float(), ref_q.float(), atol=4.0, rtol=0.0)
    check_error(LIBINFINIOP.infiniopDestroyDsv4ActQuantFp8Descriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 act_quant_fp8 Test passed!\033[0m")
