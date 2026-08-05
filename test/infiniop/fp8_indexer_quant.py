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

LIBINFINIOP.infiniopCreateFp8IndexerQuantDescriptor.restype = c_int32
LIBINFINIOP.infiniopCreateFp8IndexerQuantDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopOperatorDescriptor_t),
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
]
LIBINFINIOP.infiniopFp8IndexerQuant.restype = c_int32
LIBINFINIOP.infiniopFp8IndexerQuant.argtypes = [
    infiniopOperatorDescriptor_t,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
]
LIBINFINIOP.infiniopDestroyFp8IndexerQuantDescriptor.restype = c_int32
LIBINFINIOP.infiniopDestroyFp8IndexerQuantDescriptor.argtypes = [
    infiniopOperatorDescriptor_t
]

_TEST_CASES = [
    ((2, 3, 128),),
    ((1, 5, 96),),
    ((3, 2, 257),),
]
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]


def test(handle, device, shape, dtype, sync):
    print(
        f"Testing Fp8IndexerQuant on {InfiniDeviceNames[device]} with "
        f"shape={shape}, dtype={InfiniDtypeNames[dtype]}"
    )
    q = TestTensor(shape, None, dtype, device, scale=8.0, bias=-4.0)
    weights = TestTensor(shape[:2], None, dtype, device, scale=2.0, bias=-1.0)
    q_fp8 = TestTensor(shape, None, InfiniDtype.F8, device, mode="zeros")
    weights_fp32 = TestTensor(shape[:2], None, InfiniDtype.F32, device, mode="zeros")

    q_float = q.torch_tensor().float()
    scales = torch.exp2(
        torch.ceil(torch.log2(q_float.abs().amax(dim=-1).clamp_min(1.0e-4) / 448.0))
    )
    expected_q = (
        (q_float / scales.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    expected_weights = weights.torch_tensor().float() * scales

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFp8IndexerQuantDescriptor(
            handle,
            ctypes.byref(descriptor),
            q_fp8.descriptor,
            weights_fp32.descriptor,
            q.descriptor,
            weights.descriptor,
        )
    )
    for tensor in (q_fp8, weights_fp32, q, weights):
        tensor.destroy_desc()

    check_error(
        LIBINFINIOP.infiniopFp8IndexerQuant(
            descriptor,
            q_fp8.data(),
            weights_fp32.data(),
            q.data(),
            weights.data(),
            None,
        )
    )
    if sync is not None:
        sync()

    assert torch.equal(
        q_fp8.actual_tensor().view(torch.uint8),
        expected_q.view(torch.uint8),
    )
    assert torch.allclose(
        weights_fp32.actual_tensor(), expected_weights, atol=1.0e-7, rtol=1.0e-6
    )
    check_error(LIBINFINIOP.infiniopDestroyFp8IndexerQuantDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(0)
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mFp8IndexerQuant test passed!\033[0m")
