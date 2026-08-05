import ctypes
from ctypes import POINTER, c_double, c_int32, c_void_p

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

LIBINFINIOP.infiniopCreateFp8MlaRmsnormCacheDescriptor.restype = c_int32
LIBINFINIOP.infiniopCreateFp8MlaRmsnormCacheDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopOperatorDescriptor_t),
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    c_double,
]
LIBINFINIOP.infiniopFp8MlaRmsnormCache.restype = c_int32
LIBINFINIOP.infiniopFp8MlaRmsnormCache.argtypes = [
    infiniopOperatorDescriptor_t,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
]
LIBINFINIOP.infiniopDestroyFp8MlaRmsnormCacheDescriptor.restype = c_int32
LIBINFINIOP.infiniopDestroyFp8MlaRmsnormCacheDescriptor.argtypes = [
    infiniopOperatorDescriptor_t
]

_LATENT_DIM = 512
_GROUP_SIZE = 128
_NUM_GROUPS = 4
_ROPE_DIM = 64
_CACHE_STRIDE = _LATENT_DIM + _NUM_GROUPS * 4 + _ROPE_DIM * 2
_EPS = 1.0e-6
_TEST_CASES = [(3, True), (2, False)]
_TENSOR_DTYPES = [None]


def test(handle, device, num_tokens, write_vendor_cache, _dtype, sync):
    print(
        f"Testing Fp8MlaRmsnormCache on {InfiniDeviceNames[device]} with "
        f"tokens={num_tokens}, vendor_cache={write_vendor_cache}"
    )
    num_blocks, block_size = 2, 4
    compressed_src = (torch.rand(num_tokens, _LATENT_DIM) * 4.0 - 2.0).to(
        torch.bfloat16
    )
    weight_src = (torch.rand(_LATENT_DIM) + 0.5).to(torch.bfloat16)
    rope_src = (torch.rand(num_tokens, _ROPE_DIM) * 2.0 - 1.0).to(torch.bfloat16)
    slots_src = torch.tensor(
        [0, block_size + 1] + ([-1] if num_tokens == 3 else []),
        dtype=torch.int64,
    )

    cache = TestTensor(
        (num_blocks, block_size, _CACHE_STRIDE),
        None,
        InfiniDtype.U8,
        device,
        mode="zeros",
    )
    vendor_cache = (
        TestTensor(
            (num_blocks, block_size, _LATENT_DIM + _ROPE_DIM),
            None,
            InfiniDtype.BF16,
            device,
            mode="zeros",
        )
        if write_vendor_cache
        else None
    )
    compressed = TestTensor.from_torch(compressed_src, InfiniDtype.BF16, device)
    weight = TestTensor.from_torch(weight_src, InfiniDtype.BF16, device)
    rope = TestTensor.from_torch(rope_src, InfiniDtype.BF16, device)
    slots = TestTensor.from_torch(slots_src, InfiniDtype.I64, device)

    expected_cache = torch.zeros(
        (num_blocks * block_size, _CACHE_STRIDE), dtype=torch.uint8
    )
    expected_vendor = torch.zeros(
        (num_blocks * block_size, _LATENT_DIM + _ROPE_DIM),
        dtype=torch.bfloat16,
    )
    for token, slot in enumerate(slots_src.tolist()):
        if slot < 0:
            continue
        value = compressed_src[token].float()
        inv_rms = torch.rsqrt(value.square().mean() + _EPS)
        normalized = (value * inv_rms * weight_src.float()).to(torch.bfloat16)
        grouped = normalized.float().reshape(_NUM_GROUPS, _GROUP_SIZE)
        scales = grouped.abs().amax(dim=-1) / 448.0
        quantized = torch.where(
            scales[:, None] > 0,
            (grouped / scales[:, None]).clamp(-448.0, 448.0),
            torch.zeros_like(grouped),
        ).to(torch.float8_e4m3fn)
        expected_cache[slot, :_LATENT_DIM] = quantized.view(torch.uint8).reshape(-1)
        scale_offset = _LATENT_DIM
        expected_cache[slot, scale_offset : scale_offset + 16] = (
            scales.contiguous().view(torch.uint8)
        )
        rope_offset = _LATENT_DIM + 16
        expected_cache[slot, rope_offset:] = rope_src[token].view(torch.uint8)
        expected_vendor[slot, :_LATENT_DIM] = (
            (quantized.float() * scales[:, None]).reshape(-1).to(torch.bfloat16)
        )
        expected_vendor[slot, _LATENT_DIM:] = rope_src[token]

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFp8MlaRmsnormCacheDescriptor(
            handle,
            ctypes.byref(descriptor),
            cache.descriptor,
            vendor_cache.descriptor if vendor_cache is not None else None,
            compressed.descriptor,
            weight.descriptor,
            rope.descriptor,
            slots.descriptor,
            _EPS,
        )
    )
    tensors = [cache, compressed, weight, rope, slots]
    if vendor_cache is not None:
        tensors.append(vendor_cache)
    for tensor in tensors:
        tensor.destroy_desc()

    check_error(
        LIBINFINIOP.infiniopFp8MlaRmsnormCache(
            descriptor,
            cache.data(),
            vendor_cache.data() if vendor_cache is not None else None,
            compressed.data(),
            weight.data(),
            rope.data(),
            slots.data(),
            None,
        )
    )
    if sync is not None:
        sync()

    actual_cache = cache.actual_tensor().cpu().reshape(-1, _CACHE_STRIDE)
    assert torch.equal(actual_cache, expected_cache)
    if vendor_cache is not None:
        assert torch.equal(
            vendor_cache.actual_tensor().cpu().reshape(-1, _LATENT_DIM + _ROPE_DIM),
            expected_vendor,
        )
    check_error(LIBINFINIOP.infiniopDestroyFp8MlaRmsnormCacheDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(2)
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mFp8MlaRmsnormCache test passed!\033[0m")
