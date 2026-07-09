import ctypes
from ctypes import c_uint64

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

_TEST_CASES = [((4, 32, 0),), ((4, 32, 2),), ((4, 32, 4),), ((16, 8, 7),)]


def _workspace(descriptor, getter, device):
    size = c_uint64(0)
    check_error(getter(descriptor, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


def test_mask_topk_ids(handle, device, case, dtype=InfiniDtype.I32, sync=None):
    batch, topk, num_non_padded = case
    print(
        f"Testing DSV4 mask_topk_ids on {InfiniDeviceNames[device]} "
        f"batch:{batch} topk:{topk} num_non_padded:{num_non_padded}"
    )
    topk_ids = TestTensor(
        (batch, topk), None, InfiniDtype.I32, device, randint_low=0, randint_high=256
    )
    num = TestTensor((), None, InfiniDtype.I32, device, mode="zeros")
    num.set_tensor(torch.tensor(num_non_padded, dtype=torch.int32))
    ref = topk_ids.torch_tensor().clone()
    ref[num_non_padded:] = -1
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4MaskTopkIdsDescriptor(
            handle, ctypes.byref(desc), topk_ids.descriptor, num.descriptor
        )
    )
    for tensor in [topk_ids, num]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4MaskTopkIdsWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4MaskTopkIds(
            desc, workspace.data(), workspace_size, topk_ids.data(), num.data(), None
        )
    )
    assert torch.equal(topk_ids.actual_tensor().cpu(), ref.cpu())
    check_error(LIBINFINIOP.infiniopDestroyDsv4MaskTopkIdsDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_mask_topk_ids, _TEST_CASES, [InfiniDtype.I32])
    print("\033[92mDSV4 mask_topk_ids Test passed!\033[0m")
