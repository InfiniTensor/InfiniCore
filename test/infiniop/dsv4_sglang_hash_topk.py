import ctypes
from ctypes import c_float, c_size_t

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

_CASES = [((4, 16, 6, 1),)]


def _workspace(desc, getter, device):
    size = c_size_t(0)
    check_error(getter(desc, ctypes.byref(size)))
    return TestWorkspace(size.value, device), size.value


def test_op(handle, device, shape, dtype=InfiniDtype.F32, sync=None):
    num_tokens, num_experts, topk, shared = shape
    routed_scaling_factor = 2.0
    print(f"Testing DSV4 sglang_hash_topk on {InfiniDeviceNames[device]} shape:{shape}")
    torch.manual_seed(11)
    router_logits_torch = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda"
    ).contiguous()
    input_ids_torch = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    tid2eid_torch = (
        torch.arange(topk, dtype=torch.int32, device="cuda")
        .repeat(num_tokens, 1)
        .contiguous()
    )

    router_logits = TestTensor.from_torch(router_logits_torch, InfiniDtype.F32, device)
    input_ids = TestTensor.from_torch(input_ids_torch, InfiniDtype.I64, device)
    tid2eid = TestTensor.from_torch(tid2eid_torch, InfiniDtype.I32, device)
    topk_weights = TestTensor(
        (num_tokens, topk + shared), None, InfiniDtype.F32, device, mode="zeros"
    )
    topk_ids = TestTensor(
        (num_tokens, topk + shared), None, InfiniDtype.I32, device, mode="zeros"
    )

    if sync:
        sync()
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDsv4SglangHashTopkDescriptor(
            handle,
            ctypes.byref(desc),
            router_logits.descriptor,
            input_ids.descriptor,
            tid2eid.descriptor,
            topk_weights.descriptor,
            topk_ids.descriptor,
            c_float(routed_scaling_factor),
        )
    )
    for tensor in [router_logits, input_ids, tid2eid, topk_weights, topk_ids]:
        tensor.destroy_desc()
    workspace, workspace_size = _workspace(
        desc,
        LIBINFINIOP.infiniopGetDsv4SglangHashTopkWorkspaceSize,
        device,
    )
    check_error(
        LIBINFINIOP.infiniopDsv4SglangHashTopk(
            desc,
            workspace.data(),
            workspace_size,
            router_logits.data(),
            input_ids.data(),
            tid2eid.data(),
            topk_weights.data(),
            topk_ids.data(),
            None,
        )
    )

    routed_logits = router_logits_torch[:, :topk]
    routed_weights = torch.sqrt(torch.nn.functional.softplus(routed_logits))
    routed_weights = routed_weights / routed_weights.sum(dim=-1, keepdim=True)
    expected_ids = torch.empty(
        num_tokens, topk + shared, dtype=torch.int32, device="cuda"
    )
    expected_ids[:, :topk] = tid2eid_torch
    expected_ids[:, topk:] = num_experts
    expected_weights = torch.empty(
        num_tokens, topk + shared, dtype=torch.float32, device="cuda"
    )
    expected_weights[:, :topk] = routed_weights
    expected_weights[:, topk:] = 1.0 / routed_scaling_factor

    assert torch.equal(topk_ids.actual_tensor(), expected_ids)
    assert torch.allclose(
        topk_weights.actual_tensor(), expected_weights, atol=1e-5, rtol=1e-5
    ), (
        "DSV4 sglang_hash_topk weight mismatch: "
        f"maxdiff={(topk_weights.actual_tensor() - expected_weights).abs().max().item():.6f}"
    )
    check_error(LIBINFINIOP.infiniopDestroyDsv4SglangHashTopkDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_op, _CASES, [InfiniDtype.F32])
    print("\033[92mDSV4 sglang_hash_topk Test passed!\033[0m")
