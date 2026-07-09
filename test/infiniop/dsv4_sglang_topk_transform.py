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

_CASES = [(512,), (1024,)]


def _workspace(desc, getter, device):
    size = c_size_t(0)
    check_error(getter(desc, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


def test_op(handle, device, topk_width, dtype=InfiniDtype.F32, sync=None):
    print(
        f"Testing DSV4 sglang_topk_transform on {InfiniDeviceNames[device]} width:{topk_width}"
    )
    batch, n_valid, page_size = 2, 128, 64
    n_scores = 64 * topk_width
    n_pages = (n_scores + page_size - 1) // page_size
    scores_torch = torch.randn(
        batch, n_scores, dtype=torch.float32, device="cuda"
    ).contiguous()
    seq_lens_torch = torch.full((batch,), n_valid, dtype=torch.int32, device="cuda")
    page_tables_torch = (
        torch.arange(n_pages, dtype=torch.int32, device="cuda")
        .repeat(batch, 1)
        .contiguous()
    )

    scores = TestTensor.from_torch(scores_torch, InfiniDtype.F32, device)
    seq_lens = TestTensor.from_torch(seq_lens_torch, InfiniDtype.I32, device)
    page_tables = TestTensor.from_torch(page_tables_torch, InfiniDtype.I32, device)
    page_indices = TestTensor(
        (batch, topk_width), None, InfiniDtype.I32, device, mode="zeros"
    )
    raw_indices = TestTensor(
        (batch, topk_width), None, InfiniDtype.I32, device, mode="zeros"
    )

    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangTopkTransformDescriptor(
            handle,
            ctypes.byref(desc),
            scores.descriptor,
            seq_lens.descriptor,
            page_tables.descriptor,
            page_indices.descriptor,
            raw_indices.descriptor,
            c_int64(page_size),
        )
    )
    for tensor in [scores, seq_lens, page_tables, page_indices, raw_indices]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc,
        LIBINFINIOP.infiniopGetDsv4SglangTopkTransformWorkspaceSize,
        device,
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangTopkTransform(
            desc,
            workspace.data(),
            workspace_size,
            scores.data(),
            seq_lens.data(),
            page_tables.data(),
            page_indices.data(),
            raw_indices.data(),
            None,
        )
    )

    expected_valid = torch.arange(n_valid, dtype=torch.int32, device="cuda")
    assert torch.equal(
        page_indices.actual_tensor()[:, :n_valid], expected_valid.expand(batch, -1)
    )
    assert torch.equal(
        raw_indices.actual_tensor()[:, :n_valid], expected_valid.expand(batch, -1)
    )
    assert (page_indices.actual_tensor()[:, n_valid:] == -1).all()
    assert (raw_indices.actual_tensor()[:, n_valid:] == -1).all()
    check_error(LIBINFINIOP.infiniopDestroyDsv4SglangTopkTransformDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.F32])
    print("\033[92mDSV4 sglang_topk_transform Test passed!\033[0m")
