import ctypes
import os
from ctypes import c_size_t
from functools import lru_cache

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


@lru_cache(maxsize=1)
def _deepseek_ops():
    path = os.environ.get("DEEPSEEK_V4_OPS_SO")
    assert path, "set DEEPSEEK_V4_OPS_SO to libdeepseek_v4_ops.so"
    import tvm_ffi

    if hasattr(tvm_ffi, "load_module"):
        return tvm_ffi.load_module(path)
    from tvm_ffi.module import load_module

    return load_module(path)


def _func(name):
    mod = _deepseek_ops()
    fn = getattr(mod, name, None)
    return fn if fn is not None else mod[name]


def _paged_inputs(batch, seq_len, max_tokens_per_req=512, swa_page_size=256):
    req_pool_indices = torch.arange(batch, dtype=torch.int64, device="cuda")
    req_to_token = torch.zeros(
        (batch, max_tokens_per_req), dtype=torch.int32, device="cuda"
    )
    for b in range(batch):
        req_to_token[b, :max_tokens_per_req] = torch.arange(
            b * max_tokens_per_req,
            (b + 1) * max_tokens_per_req,
            dtype=torch.int32,
            device="cuda",
        )
    full_to_swa = torch.arange(
        batch * max_tokens_per_req, dtype=torch.int64, device="cuda"
    )
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int64)
    extend_lens = torch.full((batch,), seq_len, dtype=torch.int64)
    return req_pool_indices, req_to_token, full_to_swa, seq_lens, extend_lens


def _ref_compress_c4(kv_score_input, ape, p, head_dim):
    if p < 7:
        kv_overlap = torch.zeros(
            4, head_dim, dtype=torch.float64, device=kv_score_input.device
        )
        score_overlap = torch.full(
            (4, head_dim),
            float("-inf"),
            dtype=torch.float64,
            device=kv_score_input.device,
        )
    else:
        kv_overlap = kv_score_input[p - 7 : p - 3, :head_dim].double()
        score_overlap = kv_score_input[
            p - 7 : p - 3, 2 * head_dim : 3 * head_dim
        ].double()
    kv_fresh = kv_score_input[p - 3 : p + 1, head_dim : 2 * head_dim].double()
    score_fresh = kv_score_input[p - 3 : p + 1, 3 * head_dim :].double()
    kv = torch.cat([kv_overlap, kv_fresh], dim=0)
    score = torch.cat([score_overlap, score_fresh], dim=0) + ape.double()
    return (kv * score.softmax(dim=0)).sum(dim=0).float()


def _ref_compress_c128(kv_score_input, ape, p, head_dim):
    lo = p - 127
    kv = kv_score_input[lo : p + 1, :head_dim].double()
    score = kv_score_input[lo : p + 1, head_dim:].double()
    return (kv * (score + ape.double()).softmax(dim=0)).sum(dim=0).float()


def test_op(handle, device, dtype=InfiniDtype.F32, sync=None):
    print(
        f"Testing DSV4 sglang_compress_c128_online_v2_prefill on {InfiniDeviceNames[device]}"
    )
    head_dim, ratio, seq_len = 512, 128, 128
    req_pool_indices, req_to_token, full_to_swa, seq_lens, extend_lens = _paged_inputs(
        1, seq_len
    )
    kv_input_t = torch.randn(
        seq_len, head_dim * 2, dtype=torch.float32, device="cuda"
    ).contiguous()
    ape_t = torch.randn(
        ratio, head_dim, dtype=torch.float32, device="cuda"
    ).contiguous()
    kv_buffer_t = torch.zeros(2, 1, head_dim * 3, dtype=torch.float32, device="cuda")
    plan_c_pin = torch.empty(seq_len, 16, dtype=torch.uint8, pin_memory=True)
    plan_w_pin = torch.empty(seq_len, 16, dtype=torch.uint8, pin_memory=True)
    plan_c_t = torch.empty(seq_len, 16, dtype=torch.uint8, device="cuda")
    plan_w_t = torch.empty(seq_len, 16, dtype=torch.uint8, device="cuda")
    lens = _func("sglang_compress_c128_online_v2_plan_prefill_out")(
        seq_lens,
        extend_lens,
        req_pool_indices,
        req_to_token,
        full_to_swa,
        plan_c_pin,
        plan_w_pin,
        plan_c_t,
        plan_w_t,
        256,
    )
    num_c, num_w = int(lens[0]), int(lens[1])
    assert (num_c, num_w) == (1, 0)
    out_t = torch.empty(num_c, head_dim, dtype=torch.float32, device="cuda")
    kv_buffer = TestTensor.from_torch(kv_buffer_t, dtype, device)
    kv_input = TestTensor.from_torch(kv_input_t, dtype, device)
    out = TestTensor.from_torch(out_t, dtype, device)
    ape = TestTensor.from_torch(ape_t, dtype, device)
    plan_c = TestTensor.from_torch(
        plan_c_t[:num_c].contiguous(), InfiniDtype.U8, device
    )
    plan_w = TestTensor.from_torch(
        plan_w_t[:num_w].contiguous(), InfiniDtype.U8, device
    )
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangCompressC128OnlineV2PrefillDescriptor(
            handle,
            ctypes.byref(desc),
            kv_buffer.descriptor,
            kv_input.descriptor,
            out.descriptor,
            ape.descriptor,
            plan_c.descriptor,
            plan_w.descriptor,
        )
    )
    for tensor in [kv_buffer, kv_input, out, ape, plan_c, plan_w]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc,
        LIBINFINIOP.infiniopGetDsv4SglangCompressC128OnlineV2PrefillWorkspaceSize,
        device,
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangCompressC128OnlineV2Prefill(
            desc,
            workspace.data(),
            workspace_size,
            kv_buffer.data(),
            kv_input.data(),
            out.data(),
            ape.data(),
            plan_c.data(),
            plan_w.data(),
            None,
        )
    )
    assert torch.isfinite(out.actual_tensor()).all()
    assert out.actual_tensor().abs().sum().item() > 0
    check_error(
        LIBINFINIOP.infiniopDestroyDsv4SglangCompressC128OnlineV2PrefillDescriptor(desc)
    )


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, [()], [InfiniDtype.F32])
    print("\033[92mDSV4 sglang_compress_c128_online_v2_prefill Test passed!\033[0m")
