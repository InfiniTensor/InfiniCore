"""Ascend ApplyRoPE C API: CPU golden, address reuse and input guards."""

import ctypes

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceEnum,
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

create = LIBINFINIOP.infiniopCreateApplyRoPEDescriptor
create.argtypes = [infiniopHandle_t, ctypes.POINTER(infiniopOperatorDescriptor_t)] + [
    infiniopTensorDescriptor_t
] * 4
create.restype = ctypes.c_int
workspace_size = LIBINFINIOP.infiniopGetApplyRoPEWorkspaceSize
workspace_size.argtypes = [
    infiniopOperatorDescriptor_t,
    ctypes.POINTER(ctypes.c_size_t),
]
workspace_size.restype = ctypes.c_int
run = LIBINFINIOP.infiniopApplyRoPE
run.argtypes = [infiniopOperatorDescriptor_t, ctypes.c_void_p, ctypes.c_size_t] + [
    ctypes.c_void_p
] * 5
run.restype = ctypes.c_int
destroy = LIBINFINIOP.infiniopDestroyApplyRoPEDescriptor
destroy.argtypes = [infiniopOperatorDescriptor_t]
destroy.restype = ctypes.c_int


def rotate_cpu(x, cos, sin):
    x = x.float()
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos.float() + rotated * sin.float()


def test(handle, device, tokens, heads, dim, dtype, sync):
    cpu_dtype = {
        InfiniDtype.F16: torch.float16,
        InfiniDtype.BF16: torch.bfloat16,
        InfiniDtype.F32: torch.float32,
    }[dtype]
    generator = torch.Generator().manual_seed(tokens + heads + dim)
    desc = infiniopOperatorDescriptor_t()
    # All five sets remain alive, so each call has genuinely different addresses.
    sets = []
    for step in range(5):
        q = torch.randn(tokens, heads + 1, dim, generator=generator).to(cpu_dtype)
        k = torch.randn(tokens, heads, dim, generator=generator).to(cpu_dtype)
        angle = torch.randn(tokens, 1, dim // 2, generator=generator)
        cos = torch.cat((angle.cos(), angle.cos()), dim=-1).to(cpu_dtype)
        sin = torch.cat((angle.sin(), angle.sin()), dim=-1).to(cpu_dtype)
        tensors = [TestTensor.from_torch(x, dtype, device) for x in (q, k, cos, sin)]
        sets.append((tensors, q, k, cos, sin))
    first = sets[0][0]
    if sync:
        sync()
    check_error(create(handle, ctypes.byref(desc), *[t.descriptor for t in first]))
    # Descriptor metadata must be owned independently by the operator.
    for t in first:
        t.destroy_desc()
    size = ctypes.c_size_t()
    check_error(workspace_size(desc, ctypes.byref(size)))
    workspace = TestWorkspace(size.value, device)
    try:
        for tensors, q, k, cos, sin in sets:
            if sync:
                sync()
            check_error(
                run(
                    desc,
                    workspace.data(),
                    size.value,
                    *[t.data() for t in tensors],
                    None,
                )
            )
            if sync:
                sync()
            atol = {
                InfiniDtype.F16: 0.004,
                InfiniDtype.BF16: 0.035,
                InfiniDtype.F32: 2e-6,
            }[dtype]
            for actual, source in zip(tensors[:2], (q, k)):
                expected = rotate_cpu(source, cos, sin).to(cpu_dtype).float()
                result = actual.actual_tensor().cpu().float()
                torch.testing.assert_close(
                    result,
                    expected,
                    atol=atol,
                    rtol=0.01 if dtype != InfiniDtype.F32 else 2e-6,
                )
        tensors = sets[-1][0]
        q, k, cos, sin = [t.data() for t in tensors]
        # Overlapping Q/K must be rejected before entering the vendor kernel.
        assert run(desc, workspace.data(), size.value, q, q, cos, sin, None) == 11
        assert run(desc, workspace.data(), size.value, None, k, cos, sin, None) == 4
        if size.value:
            assert (
                run(desc, workspace.data(), size.value - 1, q, k, cos, sin, None) == 13
            )
    finally:
        check_error(destroy(desc))
    print(
        f"PASS ApplyRoPE T={tokens} KV_heads={heads} D={dim} dtype={dtype} changed_addresses=5"
    )


def guards(handle, device, dtype=InfiniDtype.BF16):
    good = [
        TestTensor(shape, None, dtype, device)
        for shape in [(2, 2, 128), (2, 1, 128), (2, 1, 128), (2, 1, 128)]
    ]
    cases = [
        (0, TestTensor((2, 2, 128), (512, 128, 1), dtype, device), 12),
        (0, TestTensor((2, 2, 32), None, dtype, device), 11),
        (1, TestTensor((3, 1, 128), None, dtype, device), 11),
        (2, TestTensor((2, 2, 128), None, dtype, device), 11),
        (3, TestTensor((2, 1, 64), None, dtype, device), 11),
        (3, TestTensor((2, 1, 128), None, InfiniDtype.F32, device), 10),
    ]
    for index, invalid, expected in cases:
        tensors = list(good)
        tensors[index] = invalid
        desc = infiniopOperatorDescriptor_t()
        assert (
            create(handle, ctypes.byref(desc), *[t.descriptor for t in tensors])
            == expected
        )
        assert not bool(desc)
    assert (
        create(
            None,
            ctypes.byref(infiniopOperatorDescriptor_t()),
            *[t.descriptor for t in good],
        )
        == 4
    )
    print("PASS ApplyRoPE invalid shape/dtype/stride/null guards")


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        if device != InfiniDeviceEnum.ASCEND:
            print("ApplyRoPE currently supports Ascend only; skipping", device)
            continue
        cases = [(t, h, d) for t in (1, 41, 2048) for h in (1, 16) for d in (64, 128)]
        test_operator(
            device, test, cases, [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]
        )

        def check_guards(handle, device, dtype, sync):
            guards(handle, device)

        test_operator(device, check_guards, [()], [InfiniDtype.BF16])
    print("ApplyRoPE regression tests passed")
