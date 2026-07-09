import ctypes
from ctypes import c_int32, c_uint64

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
    size = c_uint64(0)
    check_error(getter(desc, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


_CASES = [((2, 128, 32),), ((4, 256, 128),)]


def test_op(handle, device, case, dtype=InfiniDtype.I32, sync=None):
    batch, seq_len, window = case
    print(
        f"Testing DSV4 swa_prefill_indices on {InfiniDeviceNames[device]} batch:{batch} seq:{seq_len} window:{window}"
    )
    indices = TestTensor((batch, seq_len), None, InfiniDtype.I32, device, mode="zeros")
    ref = torch.empty((batch, seq_len), dtype=torch.int32)
    for i in range(seq_len):
        ref[:, i] = max(0, i - window + 1)
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SwaPrefillIndicesDescriptor(
            handle, ctypes.byref(desc), indices.descriptor, c_int32(window)
        )
    )
    indices.destroy_desc()
    ws, wsz = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4SwaPrefillIndicesWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SwaPrefillIndices(
            desc, ws.data(), wsz, indices.data(), None
        )
    )
    assert torch.equal(indices.actual_tensor().cpu(), ref)
    check_error(LIBINFINIOP.infiniopDestroyDsv4SwaPrefillIndicesDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.I32])
    print("\033[92mDSV4 swa_prefill_indices Test passed!\033[0m")
