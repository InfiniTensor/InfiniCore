import ctypes
from ctypes import c_uint64, c_void_p

import torch
import torch.nn.functional as F
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
    torch_device_map,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================

# ntokens, hidden_size, topk, intermediate_size, num_experts, use_device_ptrs
_TEST_CASES_ = [
    (1, 8, 1, 4, 2, False),
    (2, 16, 2, 8, 4, False),
    (3, 32, 2, 16, 4, True),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-2, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


def _make_tensor(shape, dtype, device, start, scale):
    values = torch.arange(
        start, start + torch.tensor(shape).prod().item(), dtype=torch.float32
    )
    values = ((values % 17) - 8) * scale
    return TestTensor.from_torch(values.reshape(shape), dtype, device)


def _reference_deepseek_moe(
    hidden, topk_indices, topk_weights, gate_weights, up_weights, down_weights
):
    ntokens, hidden_size = hidden.shape
    topk = topk_indices.shape[1]
    out = torch.empty_like(hidden)

    for token in range(ntokens):
        token_out = torch.zeros(hidden_size, dtype=torch.float32, device=hidden.device)
        x = hidden[token].float()
        for k in range(topk):
            expert = int(topk_indices[token, k].item())
            gate = F.linear(x, gate_weights[expert].float())
            up = F.linear(x, up_weights[expert].float())
            intermediate = (F.silu(gate) * up * topk_weights[token, k]).to(hidden.dtype)
            token_out += F.linear(intermediate.float(), down_weights[expert].float())
        out[token] = token_out.to(hidden.dtype)

    return out


def _ptr_array(tensors):
    return (c_void_p * len(tensors))(*[tensor.data() for tensor in tensors])


def _device_ptr_tensor(tensors, device):
    return torch.tensor(
        [tensor.data() for tensor in tensors],
        dtype=torch.uint64,
        device=torch_device_map[device],
    )


def test(
    handle,
    device,
    ntokens,
    hidden_size,
    topk,
    intermediate_size,
    num_experts,
    use_device_ptrs,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing DeepseekMoe on {InfiniDeviceNames[device]} with "
        f"ntokens:{ntokens} hidden_size:{hidden_size} topk:{topk} "
        f"intermediate_size:{intermediate_size} num_experts:{num_experts} "
        f"use_device_ptrs:{use_device_ptrs} dtype:{InfiniDtypeNames[dtype]}"
    )

    hidden = _make_tensor((ntokens, hidden_size), dtype, device, 0, 0.02)
    topk_indices_data = (
        torch.arange(ntokens * topk, dtype=torch.int32).reshape(ntokens, topk)
        % num_experts
    )
    topk_indices = TestTensor.from_torch(topk_indices_data, InfiniDtype.I32, device)

    topk_weights_data = torch.arange(
        1, ntokens * topk + 1, dtype=torch.float32
    ).reshape(ntokens, topk)
    topk_weights_data = topk_weights_data / topk_weights_data.sum(dim=-1, keepdim=True)
    topk_weights = TestTensor.from_torch(topk_weights_data, InfiniDtype.F32, device)

    gate_weights = [
        _make_tensor(
            (intermediate_size, hidden_size), dtype, device, 100 + i * 97, 0.01
        )
        for i in range(num_experts)
    ]
    up_weights = [
        _make_tensor(
            (intermediate_size, hidden_size), dtype, device, 200 + i * 97, 0.01
        )
        for i in range(num_experts)
    ]
    down_weights = [
        _make_tensor(
            (hidden_size, intermediate_size), dtype, device, 300 + i * 97, 0.01
        )
        for i in range(num_experts)
    ]
    out = TestTensor((ntokens, hidden_size), None, dtype, device, mode="zeros")

    ans = _reference_deepseek_moe(
        hidden.torch_tensor(),
        topk_indices.torch_tensor(),
        topk_weights.torch_tensor(),
        [weight.torch_tensor() for weight in gate_weights],
        [weight.torch_tensor() for weight in up_weights],
        [weight.torch_tensor() for weight in down_weights],
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDeepseekMoeDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            hidden.descriptor,
            topk_indices.descriptor,
            topk_weights.descriptor,
            intermediate_size,
            num_experts,
        )
    )

    for tensor in [
        out,
        hidden,
        topk_indices,
        topk_weights,
        *gate_weights,
        *up_weights,
        *down_weights,
    ]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetDeepseekMoeWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, out.device)

    gate_ptrs = _ptr_array(gate_weights)
    up_ptrs = _ptr_array(up_weights)
    down_ptrs = _ptr_array(down_weights)
    gate_device_ptrs = _device_ptr_tensor(gate_weights, device)
    up_device_ptrs = _device_ptr_tensor(up_weights, device)
    down_device_ptrs = _device_ptr_tensor(down_weights, device)

    def lib_deepseek_moe():
        if use_device_ptrs:
            check_error(
                LIBINFINIOP.infiniopDeepseekMoeWithDevicePtrs(
                    descriptor,
                    workspace.data(),
                    workspace_size.value,
                    out.data(),
                    hidden.data(),
                    topk_indices.data(),
                    topk_weights.data(),
                    gate_device_ptrs.data_ptr(),
                    up_device_ptrs.data_ptr(),
                    down_device_ptrs.data_ptr(),
                    None,
                )
            )
        else:
            check_error(
                LIBINFINIOP.infiniopDeepseekMoe(
                    descriptor,
                    workspace.data(),
                    workspace_size.value,
                    out.data(),
                    hidden.data(),
                    topk_indices.data(),
                    topk_weights.data(),
                    gate_ptrs,
                    up_ptrs,
                    down_ptrs,
                    None,
                )
            )

    lib_deepseek_moe()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: _reference_deepseek_moe(
                hidden.torch_tensor(),
                topk_indices.torch_tensor(),
                topk_weights.torch_tensor(),
                [weight.torch_tensor() for weight in gate_weights],
                [weight.torch_tensor() for weight in up_weights],
                [weight.torch_tensor() for weight in down_weights],
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_deepseek_moe(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyDeepseekMoeDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
