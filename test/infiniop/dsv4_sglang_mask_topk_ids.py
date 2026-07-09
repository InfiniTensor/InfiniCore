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

_CASES = [((4, 32),)]


def _workspace(desc, getter, device):
    size = c_size_t(0)
    check_error(getter(desc, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


def test_op(handle, device, shape, dtype=InfiniDtype.I32, sync=None):
    print(
        f"Testing DSV4 sglang_mask_topk_ids on {InfiniDeviceNames[device]} shape:{shape}"
    )
    batch, topk = shape
    torch.manual_seed(7)
    ids_torch = torch.randint(0, 256, shape, dtype=torch.int32, device="cuda")
    count_torch = torch.tensor([2], dtype=torch.int32, device="cuda")
    expected = ids_torch.clone()
    expected[2:] = -1

    topk_ids = TestTensor.from_torch(ids_torch, InfiniDtype.I32, device)
    count = TestTensor.from_torch(count_torch, InfiniDtype.I32, device)

    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangMaskTopkIdsDescriptor(
            handle,
            ctypes.byref(desc),
            topk_ids.descriptor,
            count.descriptor,
        )
    )
    for tensor in [topk_ids, count]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc,
        LIBINFINIOP.infiniopGetDsv4SglangMaskTopkIdsWorkspaceSize,
        device,
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangMaskTopkIds(
            desc,
            workspace.data(),
            workspace_size,
            topk_ids.data(),
            count.data(),
            None,
        )
    )
    assert torch.equal(topk_ids.actual_tensor(), expected)
    check_error(LIBINFINIOP.infiniopDestroyDsv4SglangMaskTopkIdsDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.I32])
    print("\033[92mDSV4 sglang_mask_topk_ids Test passed!\033[0m")
