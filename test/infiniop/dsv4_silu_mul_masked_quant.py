import ctypes
from ctypes import c_size_t

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


_CASES = [((16, 2048),), ((64, 2048),)]


def _make_mask(rows, device):
    mask = torch.ones(rows, dtype=torch.int32, device="cuda")
    mask[1::3] = 0
    return mask


def test_op(handle, device, shape, dtype=InfiniDtype.BF16, sync=None):
    print(
        f"Testing DSV4 silu_mul_masked_quant on {InfiniDeviceNames[device]} shape:{shape}"
    )
    gate = TestTensor(shape, None, dtype, device)
    up = TestTensor(shape, None, dtype, device)
    q = TestTensor(shape, None, InfiniDtype.I8, device, mode="zeros")
    rows = gate.torch_tensor().numel() // shape[-1]
    scale = TestTensor((rows, 1), None, InfiniDtype.F32, device, mode="zeros")
    mask = TestTensor((rows,), None, InfiniDtype.I32, device, mode="ones")
    mask.set_tensor(_make_mask(rows, device))

    h = (
        torch.sigmoid(gate.torch_tensor().float())
        * gate.torch_tensor().float()
        * up.torch_tensor().float()
    )
    absmax = h.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    ref_q = torch.round(h * (127.0 / absmax)).clamp(-128, 127).to(torch.int8)
    ref_s = absmax / 127.0
    mask_bool = mask.torch_tensor().bool().reshape(rows, 1)
    ref_q = torch.where(
        mask_bool,
        ref_q.reshape(rows, shape[-1]),
        torch.zeros_like(ref_q.reshape(rows, shape[-1])),
    ).reshape(shape)
    ref_s = torch.where(
        mask_bool, ref_s.reshape(rows, 1), torch.zeros_like(ref_s.reshape(rows, 1))
    )

    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SiluMulMaskedQuantDescriptor(
            handle,
            ctypes.byref(desc),
            q.descriptor,
            scale.descriptor,
            gate.descriptor,
            up.descriptor,
            mask.descriptor,
        )
    )
    for tensor in [gate, up, q, scale, mask]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc,
        LIBINFINIOP.infiniopGetDsv4SiluMulMaskedQuantWorkspaceSize,
        device,
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SiluMulMaskedQuant(
            desc,
            workspace.data(),
            workspace_size,
            q.data(),
            scale.data(),
            gate.data(),
            up.data(),
            mask.data(),
            None,
        )
    )
    assert (q.actual_tensor().float() - ref_q.float()).abs().max().item() <= 1.0
    assert torch.allclose(
        scale.actual_tensor().reshape(ref_s.shape), ref_s, atol=1e-5, rtol=1e-5
    )
    check_error(LIBINFINIOP.infiniopDestroyDsv4SiluMulMaskedQuantDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 silu_mul_masked_quant Test passed!\033[0m")
