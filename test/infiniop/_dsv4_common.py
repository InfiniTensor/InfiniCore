import ctypes
from ctypes import c_float, c_int, c_uint64

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    infiniopOperatorDescriptor_t,
)

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]
_QUANT_CASES = [((4, 512),), ((64, 4096),), ((8, 256, 4096),)]
_RMS_CASES = [((1, 1, 512),), ((1, 64, 512),), ((1, 128, 512),)]
_SILU_CASES = [((16, 2048),), ((64, 2048),)]
_ROPE_CASES = [((128, 64, 64),), ((1, 64, 64),), ((64, 8, 64),)]
_TOPK_CASES = [((2, 64),), ((4, 128),)]


def _workspace(descriptor, getter, device):
    size = c_uint64(0)
    check_error(getter(descriptor, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


def test_quant(handle, device, shape, dtype=InfiniDtype.BF16, sync=None):
    print(
        f"Testing DSV4 per_token_quant_int8 on {InfiniDeviceNames[device]} shape:{shape} dtype:{InfiniDtypeNames[dtype]}"
    )
    x = TestTensor(shape, None, dtype, device)
    q = TestTensor(shape, None, InfiniDtype.I8, device, mode="zeros")
    rows = x.torch_tensor().numel() // x.torch_tensor().shape[-1]
    scale = TestTensor((rows, 1), None, InfiniDtype.F32, device, mode="zeros")

    ref_absmax = (
        x.torch_tensor().float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    )
    ref_q = (
        torch.round(x.torch_tensor().float() * (127.0 / ref_absmax))
        .clamp(-128, 127)
        .to(torch.int8)
    )
    ref_s = ref_absmax / 127.0
    if sync:
        sync()

    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4PerTokenQuantInt8Descriptor(
            handle, ctypes.byref(desc), q.descriptor, scale.descriptor, x.descriptor
        )
    )
    for tensor in [x, q, scale]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4PerTokenQuantInt8WorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4PerTokenQuantInt8(
            desc,
            workspace.data(),
            workspace_size,
            q.data(),
            scale.data(),
            x.data(),
            None,
        )
    )
    assert (q.actual_tensor().float() - ref_q.float()).abs().max().item() <= 1.0
    assert torch.allclose(
        scale.actual_tensor().reshape(ref_s.shape), ref_s, atol=1e-5, rtol=1e-5
    )
    check_error(LIBINFINIOP.infiniopDestroyDsv4PerTokenQuantInt8Descriptor(desc))


def test_rms(handle, device, shape, dtype=InfiniDtype.BF16, sync=None):
    print(
        f"Testing DSV4 rmsnorm_self on {InfiniDeviceNames[device]} shape:{shape} dtype:{InfiniDtypeNames[dtype]}"
    )
    x = TestTensor(shape, None, dtype, device)
    y = TestTensor(shape, None, dtype, device, mode="zeros")
    eps = 1e-6
    ref = x.torch_tensor() * torch.rsqrt(
        x.torch_tensor().pow(2).mean(-1, keepdim=True) + eps
    )
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4RMSNormSelfDescriptor(
            handle, ctypes.byref(desc), y.descriptor, x.descriptor, c_float(eps)
        )
    )
    for tensor in [x, y]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4RMSNormSelfWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4RMSNormSelf(
            desc, workspace.data(), workspace_size, y.data(), x.data(), None
        )
    )
    assert torch.allclose(y.actual_tensor(), ref, atol=2e-2, rtol=2e-2)
    check_error(LIBINFINIOP.infiniopDestroyDsv4RMSNormSelfDescriptor(desc))


def test_silu(handle, device, shape, dtype=InfiniDtype.BF16, sync=None):
    print(
        f"Testing DSV4 silu_and_mul on {InfiniDeviceNames[device]} shape:{shape} dtype:{InfiniDtypeNames[dtype]}"
    )
    gate = TestTensor(shape, None, dtype, device)
    up = TestTensor(shape, None, dtype, device)
    y = TestTensor(shape, None, dtype, device, mode="zeros")
    ref = (
        torch.sigmoid(gate.torch_tensor().float())
        * gate.torch_tensor()
        * up.torch_tensor()
    ).to(gate.torch_tensor().dtype)
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SiluAndMulDescriptor(
            handle, ctypes.byref(desc), y.descriptor, gate.descriptor, up.descriptor
        )
    )
    for tensor in [gate, up, y]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4SiluAndMulWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SiluAndMul(
            desc,
            workspace.data(),
            workspace_size,
            y.data(),
            gate.data(),
            up.data(),
            None,
        )
    )
    assert torch.allclose(y.actual_tensor(), ref, atol=1e-2, rtol=1e-2)
    check_error(LIBINFINIOP.infiniopDestroyDsv4SiluAndMulDescriptor(desc))


def test_rope(handle, device, shape, dtype=InfiniDtype.BF16, sync=None):
    print(
        f"Testing DSV4 fused_rope on {InfiniDeviceNames[device]} shape:{shape} dtype:{InfiniDtypeNames[dtype]}"
    )
    q = TestTensor(shape, None, dtype, device)
    rope_dim = shape[-1]
    freq_real = TestTensor((shape[0], rope_dim // 2), None, InfiniDtype.F32, device)
    freq_imag = TestTensor((shape[0], rope_dim // 2), None, InfiniDtype.F32, device)
    q_ref = q.torch_tensor().clone()
    even = q_ref[..., 0::2].float()
    odd = q_ref[..., 1::2].float()
    real = freq_real.torch_tensor().unsqueeze(-2)
    imag = freq_imag.torch_tensor().unsqueeze(-2)
    q_ref[..., 0::2] = (even * real - odd * imag).to(q_ref.dtype)
    q_ref[..., 1::2] = (even * imag + odd * real).to(q_ref.dtype)
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4FusedRopeDescriptor(
            handle,
            ctypes.byref(desc),
            q.descriptor,
            None,
            freq_real.descriptor,
            freq_imag.descriptor,
            c_int(0),
        )
    )
    for tensor in [q, freq_real, freq_imag]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4FusedRopeWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4FusedRope(
            desc,
            workspace.data(),
            workspace_size,
            q.data(),
            None,
            freq_real.data(),
            freq_imag.data(),
            None,
        )
    )
    assert torch.allclose(q.actual_tensor(), q_ref, atol=1e-2, rtol=1e-2)
    check_error(LIBINFINIOP.infiniopDestroyDsv4FusedRopeDescriptor(desc))


def test_topk(handle, device, case, dtype=InfiniDtype.F32, sync=None):
    batch, n_valid = case
    index_topk = 512
    print(
        f"Testing DSV4 topk_transform on {InfiniDeviceNames[device]} batch:{batch} n_valid:{n_valid}"
    )
    scores = TestTensor((batch, 64 * index_topk), None, InfiniDtype.F32, device)
    seq_lens = TestTensor((batch,), None, InfiniDtype.I32, device, mode="zeros")
    seq_lens.set_tensor(torch.full((batch,), n_valid, dtype=torch.int32))
    page_tables = TestTensor((batch, 16), None, InfiniDtype.I32, device, mode="zeros")
    out = TestTensor((batch, index_topk), None, InfiniDtype.I32, device, mode="zeros")
    ref = torch.full((batch, index_topk), -1, dtype=torch.int32)
    ref[:, :n_valid] = torch.arange(n_valid, dtype=torch.int32)
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4TopkTransformDescriptor(
            handle,
            ctypes.byref(desc),
            out.descriptor,
            scores.descriptor,
            seq_lens.descriptor,
            page_tables.descriptor,
            c_int(64),
        )
    )
    for tensor in [scores, seq_lens, page_tables, out]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4TopkTransformWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4TopkTransform(
            desc,
            workspace.data(),
            workspace_size,
            out.data(),
            scores.data(),
            seq_lens.data(),
            page_tables.data(),
            None,
        )
    )
    assert torch.equal(out.actual_tensor().cpu(), ref)
    check_error(LIBINFINIOP.infiniopDestroyDsv4TopkTransformDescriptor(desc))
