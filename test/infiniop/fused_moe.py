import ctypes
from ctypes import c_uint64, c_int32
import torch
import torch.nn.functional as F

from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
    torch_device_map,
)

ACT_SILU = 0
ACT_SWIGLU = 1

_TEST_CASES_ = [
    # N, hidden, inter, experts, topk, activation
    (2, 16, 32, 4, 2, ACT_SILU),
    (3, 32, 16, 5, 2, ACT_SWIGLU),
    (1, 16, 16, 3, 1, ACT_SWIGLU),
]
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 2e-2, "rtol": 2e-2},
    InfiniDtype.BF16: {"atol": 5e-2, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-4},
}

DEBUG = False


def torch_fused_moe(x, indices, scales, w1, w2, b1, b2, activation):
    N, hidden = x.shape
    topk = indices.shape[1]
    out = torch.zeros((N, hidden), dtype=torch.float32, device=x.device)
    for n in range(N):
        x_f = x[n].float()
        for k in range(topk):
            expert = int(indices[n, k].item())
            hidden1 = torch.matmul(w1[expert].float(), x_f)
            if b1 is not None:
                hidden1 = hidden1 + b1[expert].float()
            if activation == ACT_SWIGLU:
                gate, up = hidden1.chunk(2, dim=0)
                act = F.silu(gate) * up
            else:
                act = F.silu(hidden1)
            y = torch.matmul(w2[expert].float(), act)
            if b2 is not None:
                y = y + b2[expert].float()
            out[n] += scales[n, k].float() * y
    return out.to(x.dtype)


def test(handle, device, N, hidden, inter, experts, topk, activation, dtype=InfiniDtype.F16, sync=None):
    print(
        f"Testing FusedMoe on {InfiniDeviceNames[device]} N={N} hidden={hidden} inter={inter} "
        f"experts={experts} topk={topk} activation={activation} dtype={InfiniDtypeNames[dtype]}"
    )
    torch_device = torch_device_map[device]
    torch_dtype = {InfiniDtype.F16: torch.float16, InfiniDtype.BF16: torch.bfloat16, InfiniDtype.F32: torch.float32}[dtype]

    x_t = (torch.rand((N, hidden), dtype=torch_dtype, device=torch_device) * 2 - 1).contiguous()
    w1_cols = inter * 2 if activation == ACT_SWIGLU else inter
    w1_t = (torch.rand((experts, w1_cols, hidden), dtype=torch_dtype, device=torch_device) * 2 - 1).contiguous()
    w2_t = (torch.rand((experts, hidden, inter), dtype=torch_dtype, device=torch_device) * 2 - 1).contiguous()
    b1_t = (torch.rand((experts, w1_cols), dtype=torch_dtype, device=torch_device) * 0.1).contiguous()
    b2_t = (torch.rand((experts, hidden), dtype=torch_dtype, device=torch_device) * 0.1).contiguous()

    logits = torch.rand((N, experts), dtype=torch.float32, device=torch_device)
    scales_t, indices_i64 = torch.topk(F.softmax(logits, dim=-1), topk, dim=-1)
    scales_t = scales_t / scales_t.sum(dim=-1, keepdim=True)
    indices_t = indices_i64.to(torch.int32).contiguous()
    scales_t = scales_t.contiguous()

    ans = torch_fused_moe(x_t, indices_t, scales_t, w1_t, w2_t, b1_t, b2_t, activation)

    x = TestTensor.from_torch(x_t, dtype, device)
    w1 = TestTensor.from_torch(w1_t, dtype, device)
    w2 = TestTensor.from_torch(w2_t, dtype, device)
    b1 = TestTensor.from_torch(b1_t, dtype, device)
    b2 = TestTensor.from_torch(b2_t, dtype, device)
    indices = TestTensor.from_torch(indices_t, InfiniDtype.I32, device)
    scales = TestTensor.from_torch(scales_t, InfiniDtype.F32, device)
    out = TestTensor((N, hidden), None, dtype, device)

    if sync:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFusedMoeDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            x.descriptor,
            indices.descriptor,
            scales.descriptor,
            w1.descriptor,
            w2.descriptor,
            b1.descriptor,
            b2.descriptor,
            c_int32(activation),
        )
    )

    workspace_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetFusedMoeWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
    workspace = TestWorkspace(workspace_size.value, x.device)

    for tensor in [x, w1, w2, b1, b2, indices, scales, out]:
        tensor.destroy_desc()

    check_error(
        LIBINFINIOP.infiniopFusedMoe(
            descriptor,
            workspace.data(),
            workspace_size.value,
            out.data(),
            x.data(),
            indices.data(),
            scales.data(),
            w1.data(),
            w2.data(),
            b1.data(),
            b2.data(),
            None,
        )
    )
    if sync:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    check_error(LIBINFINIOP.infiniopDestroyFusedMoeDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")
