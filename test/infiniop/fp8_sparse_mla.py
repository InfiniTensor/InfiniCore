import ctypes
from ctypes import POINTER, c_float, c_int32, c_uint64, c_void_p

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
    infiniopHandle_t,
    infiniopOperatorDescriptor_t,
    infiniopTensorDescriptor_t,
    test_operator,
)

LIBINFINIOP.infiniopCreateFp8SparseMlaDescriptor.restype = c_int32
LIBINFINIOP.infiniopCreateFp8SparseMlaDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopOperatorDescriptor_t),
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    c_float,
]
LIBINFINIOP.infiniopGetFp8SparseMlaWorkspaceSize.restype = c_int32
LIBINFINIOP.infiniopGetFp8SparseMlaWorkspaceSize.argtypes = [
    infiniopOperatorDescriptor_t,
    POINTER(c_uint64),
]
LIBINFINIOP.infiniopFp8SparseMla.restype = c_int32
LIBINFINIOP.infiniopFp8SparseMla.argtypes = [
    infiniopOperatorDescriptor_t,
    c_void_p,
    c_uint64,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
]
LIBINFINIOP.infiniopDestroyFp8SparseMlaDescriptor.restype = c_int32
LIBINFINIOP.infiniopDestroyFp8SparseMlaDescriptor.argtypes = [
    infiniopOperatorDescriptor_t
]

_LATENT_DIM = 512
_ROPE_DIM = 64
_HEAD_DIM = _LATENT_DIM + _ROPE_DIM
_CACHE_STRIDE = _LATENT_DIM + 16 + _ROPE_DIM * 2
_SCALE = 0.04
_TEST_CASES = [(2, 2, 70)]
_TENSOR_DTYPES = [None]


def test(handle, device, num_tokens, num_heads, topk, _dtype, sync):
    print(
        f"Testing Fp8SparseMla on {InfiniDeviceNames[device]} with "
        f"tokens={num_tokens}, heads={num_heads}, topk={topk}"
    )
    num_cache_tokens = 80
    latent = (torch.rand(num_cache_tokens, _LATENT_DIM) * 8.0 - 4.0).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(num_cache_tokens, 4, dtype=torch.float32) * 0.02 + 0.002
    rope = (torch.rand(num_cache_tokens, _ROPE_DIM) * 2.0 - 1.0).to(torch.bfloat16)
    cache_src = torch.zeros((num_cache_tokens, 1, _CACHE_STRIDE), dtype=torch.uint8)
    flat_cache = cache_src[:, 0]
    flat_cache[:, :_LATENT_DIM] = latent.view(torch.uint8).reshape(num_cache_tokens, -1)
    flat_cache[:, _LATENT_DIM : _LATENT_DIM + 16] = (
        scales.contiguous().view(torch.uint8).reshape(num_cache_tokens, -1)
    )
    flat_cache[:, _LATENT_DIM + 16 :] = rope.view(torch.uint8).reshape(
        num_cache_tokens, -1
    )
    query_src = (torch.rand(num_tokens, num_heads, _HEAD_DIM) - 0.5).to(torch.bfloat16)
    indices_src = torch.arange(topk, dtype=torch.int32).repeat(num_tokens, 1, 1)
    indices_src[0, 0, 2] = -1
    indices_src[1, 0, 67] = num_cache_tokens
    topk_lens_src = torch.tensor([5, topk], dtype=torch.int32)

    dequantized = latent.float().reshape(num_cache_tokens, 4, 128)
    dequantized = (dequantized * scales[:, :, None]).reshape(
        num_cache_tokens, _LATENT_DIM
    )
    keys = torch.cat([dequantized, rope.float()], dim=-1)
    expected = torch.zeros((num_tokens, num_heads, _LATENT_DIM), dtype=torch.float32)
    for token in range(num_tokens):
        valid_indices = [
            int(index)
            for index in indices_src[token, 0, : int(topk_lens_src[token])]
            if 0 <= int(index) < num_cache_tokens
        ]
        for head in range(num_heads):
            logits = (keys[valid_indices] @ query_src[token, head].float()) * _SCALE
            probabilities = torch.softmax(logits, dim=0)
            expected[token, head] = (
                probabilities[:, None] * dequantized[valid_indices]
            ).sum(dim=0)

    output = TestTensor(
        (num_tokens, num_heads, _LATENT_DIM),
        None,
        InfiniDtype.BF16,
        device,
        mode="zeros",
    )
    query = TestTensor.from_torch(query_src, InfiniDtype.BF16, device)
    cache = TestTensor.from_torch(cache_src, InfiniDtype.U8, device)
    indices = TestTensor.from_torch(indices_src, InfiniDtype.I32, device)
    topk_lens = TestTensor.from_torch(topk_lens_src, InfiniDtype.I32, device)

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFp8SparseMlaDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            query.descriptor,
            cache.descriptor,
            indices.descriptor,
            topk_lens.descriptor,
            _SCALE,
        )
    )
    for tensor in (output, query, cache, indices, topk_lens):
        tensor.destroy_desc()
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetFp8SparseMlaWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output.device)
    check_error(
        LIBINFINIOP.infiniopFp8SparseMla(
            descriptor,
            workspace.data(),
            workspace_size.value,
            output.data(),
            query.data(),
            cache.data(),
            indices.data(),
            topk_lens.data(),
            None,
        )
    )
    if sync is not None:
        sync()

    assert torch.allclose(
        output.actual_tensor().cpu().float(),
        expected,
        atol=3.0e-2,
        rtol=2.0e-2,
    )
    check_error(LIBINFINIOP.infiniopDestroyFp8SparseMlaDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(3)
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mFp8SparseMla test passed!\033[0m")
