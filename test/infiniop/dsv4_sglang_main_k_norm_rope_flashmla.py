import ctypes
from ctypes import c_double, c_size_t

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


_EPS = 1e-6
_CASES = [(5,)]


def test_op(handle, device, tokens, dtype=InfiniDtype.BF16, sync=None):
    print(
        f"Testing DSV4 sglang_main_k_norm_rope_flashmla on {InfiniDeviceNames[device]} tokens:{tokens}"
    )
    head_dim, rope_dim, page_size = 512, 64, 64
    page_bytes = 37440
    kv_torch = torch.randn(
        tokens, head_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    weight_torch = torch.randn(
        head_dim, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    positions_torch = torch.tensor([0, 3, 7, 11, 13], dtype=torch.int32, device="cuda")[
        :tokens
    ]
    out_loc_torch = torch.tensor(
        [0, 7, 65, 130, 191], dtype=torch.int32, device="cuda"
    )[:tokens]
    freqs_cis = torch.randn(16, rope_dim // 2, dtype=torch.complex64, device="cuda")
    freqs_torch = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    num_pages = int(out_loc_torch.max().item()) // page_size + 1
    kv = TestTensor.from_torch(kv_torch, dtype, device)
    weight = TestTensor.from_torch(weight_torch, dtype, device)
    freqs = TestTensor.from_torch(freqs_torch, InfiniDtype.F32, device)
    positions = TestTensor.from_torch(positions_torch, InfiniDtype.I32, device)
    out_loc = TestTensor.from_torch(out_loc_torch, InfiniDtype.I32, device)
    cache = TestTensor(
        (num_pages, page_bytes), None, InfiniDtype.U8, device, mode="zeros"
    )
    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangMainKNormRopeFlashmlaDescriptor(
            handle,
            ctypes.byref(desc),
            kv.descriptor,
            weight.descriptor,
            freqs.descriptor,
            positions.descriptor,
            out_loc.descriptor,
            cache.descriptor,
            c_double(_EPS),
        )
    )
    for tensor in [kv, weight, freqs, positions, out_loc, cache]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc,
        LIBINFINIOP.infiniopGetDsv4SglangMainKNormRopeFlashmlaWorkspaceSize,
        device,
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangMainKNormRopeFlashmla(
            desc,
            workspace.data(),
            workspace_size,
            kv.data(),
            weight.data(),
            freqs.data(),
            positions.data(),
            out_loc.data(),
            cache.data(),
            None,
        )
    )
    assert cache.actual_tensor().abs().sum().item() > 0
    check_error(
        LIBINFINIOP.infiniopDestroyDsv4SglangMainKNormRopeFlashmlaDescriptor(desc)
    )


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.BF16])
    print("\033[92mDSV4 sglang_main_k_norm_rope_flashmla Test passed!\033[0m")
