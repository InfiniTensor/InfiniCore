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


_CASES = [(3, 5, 128, 4)]


def _unpack_ue8m0_scales(buf_x_sf, num_groups):
    raw = buf_x_sf.contiguous().view(torch.uint8).reshape(buf_x_sf.shape[0], num_groups)
    return torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=buf_x_sf.device),
        raw.to(torch.float32) - 127.0,
    )


def test_op(
    handle,
    device,
    num_tokens,
    padded_max,
    hidden,
    top_k,
    dtype=InfiniDtype.BF16,
    sync=None,
):
    print(
        f"Testing DSV4 sglang_mega_moe_pre_dispatch on {InfiniDeviceNames[device]} tokens:{num_tokens} hidden:{hidden}"
    )
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch float8_e4m3fn is required for this test")
    group_size = 32
    num_groups = hidden // group_size
    x_torch = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    topk_idx_torch = torch.tensor(
        [[0, 3, 5, 7], [2, 4, 6, 8], [1, 9, 11, 13]], dtype=torch.int32, device="cuda"
    )[:, :top_k].contiguous()
    topk_weights_torch = torch.rand(
        num_tokens, top_k, dtype=torch.float32, device="cuda"
    ).contiguous()
    x = TestTensor.from_torch(x_torch, dtype, device)
    topk_idx = TestTensor.from_torch(topk_idx_torch, InfiniDtype.I32, device)
    topk_weights = TestTensor.from_torch(topk_weights_torch, InfiniDtype.F32, device)
    buf_x = TestTensor((padded_max, hidden), None, InfiniDtype.I8, device, mode="zeros")
    buf_x_sf = TestTensor(
        (padded_max, num_groups // 4), None, InfiniDtype.I32, device, mode="zeros"
    )
    buf_topk_idx = TestTensor(
        (padded_max, top_k), None, InfiniDtype.I64, device, mode="zeros"
    )
    buf_topk_weights = TestTensor(
        (padded_max, top_k), None, InfiniDtype.F32, device, mode="zeros"
    )
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangMegaMoePreDispatchDescriptor(
            handle,
            ctypes.byref(desc),
            x.descriptor,
            topk_idx.descriptor,
            topk_weights.descriptor,
            buf_x.descriptor,
            buf_x_sf.descriptor,
            buf_topk_idx.descriptor,
            buf_topk_weights.descriptor,
        )
    )
    for tensor in [
        x,
        topk_idx,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
    ]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4SglangMegaMoePreDispatchWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangMegaMoePreDispatch(
            desc,
            workspace.data(),
            workspace_size,
            x.data(),
            topk_idx.data(),
            topk_weights.data(),
            buf_x.data(),
            buf_x_sf.data(),
            buf_topk_idx.data(),
            buf_topk_weights.data(),
            None,
        )
    )
    assert torch.equal(
        buf_topk_idx.actual_tensor()[:num_tokens], topk_idx_torch.to(torch.int64)
    )
    assert torch.allclose(
        buf_topk_weights.actual_tensor()[:num_tokens],
        topk_weights_torch,
        atol=0,
        rtol=0,
    )
    q = buf_x.actual_tensor()[:num_tokens].view(torch.float8_e4m3fn).float()
    scales = _unpack_ue8m0_scales(
        buf_x_sf.actual_tensor()[:num_tokens], num_groups
    ).repeat_interleave(group_size, dim=1)
    dequant = q * scales
    assert torch.allclose(dequant, x_torch.float(), atol=0.25, rtol=0.25)
    check_error(LIBINFINIOP.infiniopDestroyDsv4SglangMegaMoePreDispatchDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 sglang_mega_moe_pre_dispatch Test passed!\033[0m")
