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


_CASES = [([1, 256, 257, 1024], 8), ([0, 13, 511, 512, 4097], 16)]


def _ref(seq_lens, num_sm):
    split_kv = 256
    seq_lens_cpu = seq_lens.detach().cpu().to(torch.int64).view(-1)
    batch = seq_lens_cpu.numel()
    works = ((seq_lens_cpu + split_kv - 1) // split_kv).tolist()
    total_work = sum(works)
    avg = total_work // num_sm
    ret = total_work % num_sm
    out = torch.empty(num_sm + 1, 2, dtype=torch.int32)
    q = 0
    num_work = works[0] if batch else 0
    sum_work = num_work
    for i in range(num_sm + 1):
        target = i * avg + min(i, ret)
        while sum_work <= target:
            q += 1
            if q >= batch:
                break
            num_work = works[q]
            sum_work += num_work
        if q >= batch:
            out[i, 0] = batch
            out[i, 1] = 0
        else:
            out[i, 0] = q
            out[i, 1] = target - (sum_work - num_work)
    return out.to(seq_lens.device)


def test_op(handle, device, seq_lens_values, num_sm, dtype=InfiniDtype.I32, sync=None):
    print(
        f"Testing DSV4 sglang_paged_mqa_logits_metadata on {InfiniDeviceNames[device]} batch:{len(seq_lens_values)} num_sm:{num_sm}"
    )
    seq_torch = torch.tensor(seq_lens_values, dtype=torch.int32, device="cuda")
    seq_lens = TestTensor.from_torch(seq_torch, InfiniDtype.I32, device)
    metadata = TestTensor((num_sm + 1, 2), None, InfiniDtype.I32, device, mode="zeros")
    ref = _ref(seq_torch, num_sm)
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangPagedMqaLogitsMetadataDescriptor(
            handle, ctypes.byref(desc), seq_lens.descriptor, metadata.descriptor
        )
    )
    for tensor in [seq_lens, metadata]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc,
        LIBINFINIOP.infiniopGetDsv4SglangPagedMqaLogitsMetadataWorkspaceSize,
        device,
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangPagedMqaLogitsMetadata(
            desc,
            workspace.data(),
            workspace_size,
            seq_lens.data(),
            metadata.data(),
            None,
        )
    )
    assert torch.equal(metadata.actual_tensor(), ref)
    check_error(
        LIBINFINIOP.infiniopDestroyDsv4SglangPagedMqaLogitsMetadataDescriptor(desc)
    )


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.I32])
    print("\033[92mDSV4 sglang_paged_mqa_logits_metadata Test passed!\033[0m")
