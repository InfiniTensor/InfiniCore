import ctypes
from ctypes import POINTER, c_int32, c_void_p

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    check_error,
    get_args,
    get_test_devices,
    infiniopHandle_t,
    infiniopOperatorDescriptor_t,
    infiniopTensorDescriptor_t,
    test_operator,
)

LIBINFINIOP.infiniopCreateSelectLastTokenHiddenDescriptor.restype = c_int32
LIBINFINIOP.infiniopCreateSelectLastTokenHiddenDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopOperatorDescriptor_t),
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
]
LIBINFINIOP.infiniopSelectLastTokenHidden.restype = c_int32
LIBINFINIOP.infiniopSelectLastTokenHidden.argtypes = [
    infiniopOperatorDescriptor_t,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
]
LIBINFINIOP.infiniopDestroySelectLastTokenHiddenDescriptor.restype = c_int32
LIBINFINIOP.infiniopDestroySelectLastTokenHiddenDescriptor.argtypes = [
    infiniopOperatorDescriptor_t,
]


_TEST_CASES = [
    (4, 25, 64, (0, 3, 11, 18, 25)),
    (4, 2048, 6144, (0, 512, 1024, 1536, 2048)),
]
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32]


def test(handle, device, num_requests, total_tokens, hidden_size, offsets, dtype, sync):
    print(
        f"Testing SelectLastTokenHidden on {InfiniDeviceNames[device]} with "
        f"requests={num_requests}, tokens={total_tokens}, hidden={hidden_size}, "
        f"dtype={InfiniDtypeNames[dtype]}"
    )
    hidden = TestTensor((1, total_tokens, hidden_size), None, dtype, device)
    output = TestTensor(
        (1, num_requests, hidden_size), None, dtype, device, mode="zeros"
    )
    input_offsets = TestTensor.from_torch(
        torch.tensor(offsets, dtype=torch.int32), InfiniDtype.I32, device
    )
    expected_rows = torch.tensor(
        [value - 1 for value in offsets[1:]], device=hidden.torch_tensor().device
    )
    expected = (
        hidden.torch_tensor()
        .view(total_tokens, hidden_size)
        .index_select(0, expected_rows)
    )
    output.update_torch_tensor(expected.view(1, num_requests, hidden_size))

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSelectLastTokenHiddenDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            hidden.descriptor,
            input_offsets.descriptor,
        )
    )
    output.destroy_desc()
    hidden.destroy_desc()
    input_offsets.destroy_desc()

    check_error(
        LIBINFINIOP.infiniopSelectLastTokenHidden(
            descriptor,
            output.data(),
            hidden.data(),
            input_offsets.data(),
            None,
        )
    )
    if sync is not None:
        sync()
    assert torch.equal(output.actual_tensor(), output.torch_tensor())
    check_error(LIBINFINIOP.infiniopDestroySelectLastTokenHiddenDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")
