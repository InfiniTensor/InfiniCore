import ctypes
import math
from ctypes import c_bool, c_float, c_int32, c_size_t

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

PAGE_SIZE = 64
HEAD_DIM_NOPE = 512
HEAD_DIM_PE = 64
HEAD_DIM_V = 512
KV_DIM = HEAD_DIM_NOPE + HEAD_DIM_PE


def _workspace(desc, getter, device):
    size = c_size_t(0)
    check_error(getter(desc, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


def _cdiv(a, b):
    return (a + b - 1) // b


def _num_sm_parts(cache_seqlens, num_heads_per_head_k, num_heads_k):
    block_size_m = 16
    if num_heads_per_head_k > 32:
        block_size_m = 64
    elif num_heads_per_head_k > 16:
        block_size_m = 32
    props = torch.cuda.get_device_properties(cache_seqlens.device)
    parts = props.multi_processor_count * (2 if block_size_m == 16 else 1)
    return parts // num_heads_k // _cdiv(num_heads_per_head_k, block_size_m)


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    denom = max((x * x + y * y).sum().item(), 1e-12)
    return 1 - 2 * (x * y).sum().item() / denom


def _ref_mla_decode(q_nope, q_pe, blocked_k, cache_seqlens, h_q, h_kv, causal=False):
    batch, s_q, _, _ = q_nope.shape
    max_seqlen = int(cache_seqlens.max().item())
    max_seqlen_pad = _cdiv(max_seqlen, 256) * 256
    sm_scale = (q_nope.shape[-1] + q_pe.shape[-1]) ** (-0.5)

    out = torch.empty(
        batch, s_q, h_q, HEAD_DIM_V, dtype=torch.float32, device=blocked_k.device
    )
    lse_out = torch.empty(batch, h_q, s_q, dtype=torch.float32, device=blocked_k.device)
    blocked_k_flat = blocked_k.view(-1, h_kv, KV_DIM)

    for i in range(batch):
        sl = int(cache_seqlens[i].item())
        begin = i * max_seqlen_pad
        end = begin + sl

        q_full = torch.cat([q_nope[i].float(), q_pe[i].float()], dim=-1)
        key = (
            blocked_k_flat[begin:end]
            .float()
            .transpose(0, 1)
            .repeat_interleave(h_q // h_kv, dim=0)
        )
        val = (
            blocked_k_flat[begin:end, :, :HEAD_DIM_V]
            .float()
            .transpose(0, 1)
            .repeat_interleave(h_q // h_kv, dim=0)
        )

        attn = q_full.transpose(0, 1) @ key.transpose(-2, -1) * sm_scale
        if causal:
            sq, sk = attn.shape[-2], attn.shape[-1]
            attn_bias = torch.zeros(sq, sk, dtype=torch.float32, device=attn.device)
            mask = torch.ones(sq, sk, dtype=torch.bool, device=attn.device).tril(
                diagonal=sk - sq
            )
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn = attn + attn_bias

        lse = attn.logsumexp(dim=-1)
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32)
        out[i] = (attn @ val).transpose(0, 1)
        lse_out[i] = lse

    return out.to(q_nope.dtype), lse_out


def _create_metadata(handle, device, cache_seqlens, num_heads_per_head_k, num_heads_k):
    rows = _num_sm_parts(cache_seqlens, num_heads_per_head_k, num_heads_k)
    cache_seqlens_t = TestTensor.from_torch(cache_seqlens, InfiniDtype.I32, device)
    metadata_t = TestTensor((rows, 8), None, InfiniDtype.I32, device, mode="zeros")
    num_splits_t = TestTensor(
        (cache_seqlens.numel() + 1,), None, InfiniDtype.I32, device, mode="zeros"
    )

    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4FlashMlaMetadataDescriptor(
            handle,
            ctypes.byref(desc),
            cache_seqlens_t.descriptor,
            metadata_t.descriptor,
            num_splits_t.descriptor,
            c_int32(num_heads_per_head_k),
            c_int32(num_heads_k),
        )
    )
    cache_seqlens_t.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4FlashMlaMetadataWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4FlashMlaMetadata(
            desc,
            workspace.data(),
            workspace_size,
            cache_seqlens_t.data(),
            metadata_t.data(),
            num_splits_t.data(),
            None,
        )
    )
    check_error(LIBINFINIOP.infiniopDestroyDsv4FlashMlaMetadataDescriptor(desc))
    return metadata_t, num_splits_t


def _decode(
    handle,
    device,
    q_nope,
    q_pe,
    k_cache,
    block_table,
    cache_seqlens,
    metadata_t,
    num_splits_t,
):
    q_nope_t = TestTensor.from_torch(q_nope, InfiniDtype.BF16, device)
    q_pe_t = TestTensor.from_torch(q_pe, InfiniDtype.BF16, device)
    k_cache_t = TestTensor.from_torch(k_cache, InfiniDtype.BF16, device)
    block_table_t = TestTensor.from_torch(block_table, InfiniDtype.I32, device)
    cache_seqlens_t = TestTensor.from_torch(cache_seqlens, InfiniDtype.I32, device)
    out_t = TestTensor(
        (q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], HEAD_DIM_V),
        None,
        InfiniDtype.BF16,
        device,
        mode="zeros",
    )
    lse_t = TestTensor(
        (q_nope.shape[0], q_nope.shape[2], q_nope.shape[1]),
        None,
        InfiniDtype.F32,
        device,
        mode="zeros",
    )

    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4FlashMlaDecodeDescriptor(
            handle,
            ctypes.byref(desc),
            out_t.descriptor,
            lse_t.descriptor,
            q_nope_t.descriptor,
            q_pe_t.descriptor,
            k_cache_t.descriptor,
            block_table_t.descriptor,
            cache_seqlens_t.descriptor,
            metadata_t.descriptor,
            num_splits_t.descriptor,
            c_int32(HEAD_DIM_V),
            c_float(1.0 / math.sqrt(KV_DIM)),
            c_bool(True),
        )
    )
    for tensor in [
        q_nope_t,
        q_pe_t,
        k_cache_t,
        block_table_t,
        cache_seqlens_t,
        out_t,
        lse_t,
        metadata_t,
        num_splits_t,
    ]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4FlashMlaDecodeWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4FlashMlaDecode(
            desc,
            workspace.data(),
            workspace_size,
            out_t.data(),
            lse_t.data(),
            q_nope_t.data(),
            q_pe_t.data(),
            k_cache_t.data(),
            block_table_t.data(),
            cache_seqlens_t.data(),
            metadata_t.data(),
            num_splits_t.data(),
            None,
        )
    )
    check_error(LIBINFINIOP.infiniopDestroyDsv4FlashMlaDecodeDescriptor(desc))
    return out_t.actual_tensor(), lse_t.actual_tensor()


_CASES = [(1,), (4,)]


def test_op(handle, device, batch, dtype=InfiniDtype.BF16, sync=None):
    print(f"Testing DSV4 flash_mla on {InfiniDeviceNames[device]} batch:{batch}")
    h_q = 8
    h_kv = 1
    s_q = 1
    s_k = 128
    torch.manual_seed(42 + batch)

    cache_seqlens = torch.full((batch,), s_k, dtype=torch.int32, device="cuda")
    max_seqlen_pad = _cdiv(s_k, 256) * 256
    q = torch.randn(batch, s_q, h_q, KV_DIM, dtype=torch.bfloat16, device="cuda")
    q_nope = q[..., :HEAD_DIM_NOPE].contiguous()
    q_pe = q[..., HEAD_DIM_NOPE:].contiguous()
    block_table = torch.arange(
        batch * max_seqlen_pad // PAGE_SIZE, dtype=torch.int32, device="cuda"
    ).view(batch, max_seqlen_pad // PAGE_SIZE)
    k_cache = torch.randn(
        block_table.numel(),
        PAGE_SIZE,
        h_kv,
        KV_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    for i in range(batch):
        k_cache.view(batch, max_seqlen_pad, h_kv, KV_DIM)[i, s_k:] = float("nan")

    ref_out, ref_lse = _ref_mla_decode(
        q_nope, q_pe, k_cache, cache_seqlens, h_q, h_kv, causal=True
    )
    if sync:
        sync()

    metadata_t, num_splits_t = _create_metadata(
        handle, device, cache_seqlens, s_q * h_q // h_kv, h_kv
    )
    num_splits = num_splits_t.actual_tensor()
    assert num_splits[0].item() == 0
    assert torch.all(num_splits[1:] >= num_splits[:-1])
    assert torch.count_nonzero(metadata_t.actual_tensor()).item() > 0

    out, lse = _decode(
        handle,
        device,
        q_nope,
        q_pe,
        k_cache,
        block_table,
        cache_seqlens,
        metadata_t,
        num_splits_t,
    )
    assert not torch.isnan(out).any()
    assert not torch.isnan(lse).any()
    assert _cos_diff(out, ref_out) < 1e-4
    assert _cos_diff(lse, ref_lse) < 1e-3


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 flash_mla Test passed!\033[0m")
