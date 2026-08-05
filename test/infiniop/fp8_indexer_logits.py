import ctypes
from ctypes import POINTER, c_int32, c_void_p

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    TestTensor,
    check_error,
    get_args,
    get_test_devices,
    infiniopHandle_t,
    infiniopOperatorDescriptor_t,
    infiniopTensorDescriptor_t,
    test_operator,
)

LIBINFINIOP.infiniopCreateFp8IndexerLogitsDescriptor.restype = c_int32
LIBINFINIOP.infiniopCreateFp8IndexerLogitsDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopOperatorDescriptor_t),
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
]
LIBINFINIOP.infiniopFp8IndexerLogits.restype = c_int32
LIBINFINIOP.infiniopFp8IndexerLogits.argtypes = [
    infiniopOperatorDescriptor_t,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
]
LIBINFINIOP.infiniopDestroyFp8IndexerLogitsDescriptor.restype = c_int32
LIBINFINIOP.infiniopDestroyFp8IndexerLogitsDescriptor.argtypes = [
    infiniopOperatorDescriptor_t
]

_TEST_CASES = [(2, 2), (3, 2)]
_TENSOR_DTYPES = [None]
_BLOCK_SIZE = 64
_HEAD_DIM = 128
_CACHE_STRIDE = _HEAD_DIM + 4
_MAX_CONTEXT = 128
_NUM_HEADS = 3


def _make_cache(num_blocks):
    keys = (torch.rand(num_blocks, _BLOCK_SIZE, _HEAD_DIM) * 8.0 - 4.0).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(num_blocks, _BLOCK_SIZE, dtype=torch.float32) * 0.02 + 0.001
    raw = torch.zeros((num_blocks, _BLOCK_SIZE * _CACHE_STRIDE), dtype=torch.uint8)
    raw[:, : _BLOCK_SIZE * _HEAD_DIM] = keys.view(torch.uint8).reshape(num_blocks, -1)
    raw[:, _BLOCK_SIZE * _HEAD_DIM :] = (
        scales.contiguous().view(torch.uint8).reshape(num_blocks, -1)
    )
    return raw.reshape(num_blocks, _BLOCK_SIZE, _CACHE_STRIDE), keys, scales


def test(handle, device, num_tokens, num_requests, _dtype, sync):
    print(
        f"Testing Fp8IndexerLogits on {InfiniDeviceNames[device]} with "
        f"tokens={num_tokens}, requests={num_requests}"
    )
    num_blocks = num_requests * 2
    q_src = (torch.rand(num_tokens, _NUM_HEADS, _HEAD_DIM) * 8.0 - 4.0).to(
        torch.float8_e4m3fn
    )
    weights_src = torch.rand(num_tokens, _NUM_HEADS, dtype=torch.float32)
    cache_src, keys, key_scales = _make_cache(num_blocks)
    block_tables_src = torch.arange(num_blocks, dtype=torch.int32).reshape(
        num_requests, 2
    )
    positions_src = torch.tensor([70, 45] + [12] * (num_tokens - 2), dtype=torch.int64)
    request_ids_src = torch.tensor(
        list(range(num_requests)) + [-1] * (num_tokens - num_requests),
        dtype=torch.int32,
    )

    expected = torch.full((num_tokens, _MAX_CONTEXT), -torch.inf, dtype=torch.float32)
    for token in range(num_tokens):
        request = int(request_ids_src[token])
        if request < 0 or request >= num_requests:
            continue
        for key_position in range(int(positions_src[token]) + 1):
            logical_block = key_position // _BLOCK_SIZE
            key_offset = key_position % _BLOCK_SIZE
            physical_block = int(block_tables_src[request, logical_block])
            key = keys[physical_block, key_offset].float()
            scale = key_scales[physical_block, key_offset]
            acc = 0.0
            for head in range(_NUM_HEADS):
                dot = torch.dot(q_src[token, head].float(), key)
                acc += float(torch.relu(dot * scale) * weights_src[token, head])
            expected[token, key_position] = acc

    logits = TestTensor(
        (num_tokens, _MAX_CONTEXT),
        None,
        InfiniDtype.F32,
        device,
        mode="zeros",
    )
    # q_src is contiguous. Avoid from_torch's generic strided rearrangement:
    # some device runtimes cannot round-trip FP8 through the float64 scratch
    # tensor used by that path and silently replace the input with zeros.
    q_fp8 = TestTensor(
        q_src.shape, None, InfiniDtype.F8, device, mode="manual", set_tensor=q_src
    )
    kv_cache = TestTensor.from_torch(cache_src, InfiniDtype.U8, device)
    block_tables = TestTensor.from_torch(block_tables_src, InfiniDtype.I32, device)
    weights = TestTensor.from_torch(weights_src, InfiniDtype.F32, device)
    positions = TestTensor.from_torch(positions_src, InfiniDtype.I64, device)
    request_ids = TestTensor.from_torch(request_ids_src, InfiniDtype.I32, device)

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFp8IndexerLogitsDescriptor(
            handle,
            ctypes.byref(descriptor),
            logits.descriptor,
            q_fp8.descriptor,
            kv_cache.descriptor,
            block_tables.descriptor,
            weights.descriptor,
            positions.descriptor,
            request_ids.descriptor,
        )
    )
    for tensor in (
        logits,
        q_fp8,
        kv_cache,
        block_tables,
        weights,
        positions,
        request_ids,
    ):
        tensor.destroy_desc()

    check_error(
        LIBINFINIOP.infiniopFp8IndexerLogits(
            descriptor,
            logits.data(),
            q_fp8.data(),
            kv_cache.data(),
            block_tables.data(),
            weights.data(),
            positions.data(),
            request_ids.data(),
            None,
        )
    )
    if sync is not None:
        sync()

    actual = logits.actual_tensor().cpu()
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    assert torch.allclose(actual[finite], expected[finite], atol=2.0e-3, rtol=2.0e-4)
    check_error(LIBINFINIOP.infiniopDestroyFp8IndexerLogitsDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(1)
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mFp8IndexerLogits test passed!\033[0m")
