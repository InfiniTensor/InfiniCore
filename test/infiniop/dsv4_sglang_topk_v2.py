import ctypes
from ctypes import c_int64, c_size_t

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


_CASES = [([17, 128, 511],)]


def _ref(seq_lens, page_tables, topk_width, page_size):
    out = torch.full(
        (seq_lens.shape[0], topk_width), -1, dtype=torch.int32, device=seq_lens.device
    )
    for b in range(seq_lens.shape[0]):
        n_valid = min(int(seq_lens[b].item()), topk_width)
        raw = torch.arange(n_valid, dtype=torch.int32, device=seq_lens.device)
        pages = page_tables[b, raw // page_size]
        out[b, :n_valid] = pages * page_size + raw % page_size
    return out


def test_op(handle, device, seq_values, dtype=InfiniDtype.F32, sync=None):
    print(
        f"Testing DSV4 sglang_topk_v2 on {InfiniDeviceNames[device]} batch:{len(seq_values)}"
    )
    batch, topk_width, page_size = len(seq_values), 512, 64
    seq_torch = torch.tensor(seq_values, dtype=torch.int32, device="cuda")
    scores_torch = torch.randn(
        batch, topk_width, dtype=torch.float32, device="cuda"
    ).contiguous()
    page_tables_torch = (
        torch.arange(8, dtype=torch.int32, device="cuda")
        .flip(0)
        .repeat(batch, 1)
        .contiguous()
    )
    scores = TestTensor.from_torch(scores_torch, InfiniDtype.F32, device)
    seq_lens = TestTensor.from_torch(seq_torch, InfiniDtype.I32, device)
    page_tables = TestTensor.from_torch(page_tables_torch, InfiniDtype.I32, device)
    page_indices = TestTensor(
        (batch, topk_width), None, InfiniDtype.I32, device, mode="zeros"
    )
    transform_workspace = TestTensor(
        (batch, 2 + 1024 * 2), None, InfiniDtype.I32, device, mode="zeros"
    )
    metadata = TestTensor((batch + 1, 4), None, InfiniDtype.I32, device, mode="zeros")
    ref = _ref(seq_torch, page_tables_torch, topk_width, page_size)
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangTopkV2Descriptor(
            handle,
            ctypes.byref(desc),
            scores.descriptor,
            seq_lens.descriptor,
            page_tables.descriptor,
            page_indices.descriptor,
            transform_workspace.descriptor,
            metadata.descriptor,
            c_int64(page_size),
        )
    )
    for tensor in [
        scores,
        seq_lens,
        page_tables,
        page_indices,
        transform_workspace,
        metadata,
    ]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4SglangTopkV2WorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangTopkV2(
            desc,
            workspace.data(),
            workspace_size,
            scores.data(),
            seq_lens.data(),
            page_tables.data(),
            page_indices.data(),
            transform_workspace.data(),
            metadata.data(),
            None,
        )
    )
    assert torch.equal(page_indices.actual_tensor(), ref)
    check_error(LIBINFINIOP.infiniopDestroyDsv4SglangTopkV2Descriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.F32])
    print("\033[92mDSV4 sglang_topk_v2 Test passed!\033[0m")
