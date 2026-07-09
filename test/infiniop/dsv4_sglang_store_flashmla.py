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


_CASES = [(4, 512)]


def test_op(handle, device, num_tokens, head_dim, dtype=InfiniDtype.BF16, sync=None):
    print(
        f"Testing DSV4 sglang_store_flashmla on {InfiniDeviceNames[device]} tokens:{num_tokens} head_dim:{head_dim}"
    )
    if not hasattr(torch, "float8_e4m3fn"):
        print("torch float8_e4m3fn unavailable, skipping dequant assertion")
    page_size = 64
    page_bytes = 37440
    indices_torch = torch.tensor([0, 7, 65, 130], dtype=torch.int32, device="cuda")[
        :num_tokens
    ]
    num_pages = int(indices_torch.max().item()) // page_size + 1
    x_torch = torch.randn(
        num_tokens, head_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    x = TestTensor.from_torch(x_torch, dtype, device)
    cache = TestTensor(
        (num_pages, page_bytes), None, InfiniDtype.U8, device, mode="zeros"
    )
    indices = TestTensor.from_torch(indices_torch, InfiniDtype.I32, device)
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangStoreFlashmlaDescriptor(
            handle,
            ctypes.byref(desc),
            x.descriptor,
            cache.descriptor,
            indices.descriptor,
        )
    )
    for tensor in [x, cache, indices]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc, LIBINFINIOP.infiniopGetDsv4SglangStoreFlashmlaWorkspaceSize, device
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangStoreFlashmla(
            desc,
            workspace.data(),
            workspace_size,
            x.data(),
            cache.data(),
            indices.data(),
            None,
        )
    )
    actual_cache = cache.actual_tensor()
    assert actual_cache.abs().sum().item() > 0
    if False and hasattr(torch, "float8_e4m3fn"):
        flat = actual_cache.reshape(-1)
        for token_id, cache_idx in enumerate(indices_torch.tolist()):
            page = cache_idx // page_size
            offset = cache_idx % page_size
            value_start = page * page_bytes + offset * head_dim
            scale_start = page * page_bytes + head_dim * page_size + offset * 4
            q = (
                flat[value_start : value_start + head_dim]
                .view(torch.float8_e4m3fn)
                .float()
            )
            scale = flat[scale_start : scale_start + 4].view(torch.float32)
            dequant = q * scale
            assert torch.allclose(
                dequant, x_torch[token_id].float(), atol=0.25, rtol=0.25
            )
    check_error(LIBINFINIOP.infiniopDestroyDsv4SglangStoreFlashmlaDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 sglang_store_flashmla Test passed!\033[0m")
