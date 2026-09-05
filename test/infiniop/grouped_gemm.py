import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (internal use only)
# ==============================================================================
# (alpha, beta, group_sizes, K, N)
#   a: [sum(group_sizes), K]   b: [num_groups, N, K]   c: [sum(group_sizes), N]
#
# The MoE shapes we care about for Qwen3-30B-A3B are around K=2048 N=768 E=128
# with very skewed group_sizes; we keep a small smoke set and one realistic
# shape so CI doesn't blow up.
_TEST_CASES = [
    (1.0, 0.0, [1, 1, 1, 1], 16, 8),
    (1.0, 0.0, [4, 0, 2, 7], 32, 24),
    (1.0, 1.0, [3, 5], 64, 16),
    (0.5, 0.0, [8, 8, 8, 8, 8, 8, 8, 8], 128, 64),
    # Qwen3-MoE one-layer ish: 128 experts, top-8 routing, ~256 tokens -> ~16 tokens/expert avg
    (1.0, 0.0, [16] * 128, 2048, 768),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 0, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


def torch_grouped_gemm(c_inout, a, b, group_sizes, alpha, beta):
    """Reference: per-group `linear` against the matching B slab."""
    out = torch.zeros_like(c_inout)
    offset = 0
    for g, rows in enumerate(group_sizes):
        if rows == 0:
            continue
        a_g = a[offset : offset + rows]
        b_g = b[g]  # [N, K]
        out[offset : offset + rows] = a_g @ b_g.transpose(0, 1)
        offset += rows
    return alpha * out + beta * c_inout


def test(
    handle,
    device,
    alpha,
    beta,
    group_sizes,
    k,
    n,
    dtype=InfiniDtype.F16,
    sync=None,
):
    num_groups = len(group_sizes)
    m_total = sum(group_sizes)

    print(
        f"Testing GroupedGemm on {InfiniDeviceNames[device]} with"
        f" alpha:{alpha}, beta:{beta}, groups:{num_groups},"
        f" M_total:{m_total}, K:{k}, N:{n}, dtype:{InfiniDtypeNames[dtype]}"
    )

    a = TestTensor((m_total, k), None, dtype, device)
    b = TestTensor((num_groups, n, k), None, dtype, device)
    c = TestTensor((m_total, n), None, dtype, device, mode="ones")
    ans_t = torch_grouped_gemm(
        c.torch_tensor(), a.torch_tensor(), b.torch_tensor(), group_sizes, alpha, beta
    )

    # Pass the set tensor's own stride: TestTensor's "manual" mode asserts
    # torch_strides == set_tensor.stride(), and strides=None makes torch_strides
    # None (mirrors topksoftmax.py, which passes data.stride()).
    group_sizes_data = torch.tensor(group_sizes, dtype=torch.int32)
    group_sizes_tensor = TestTensor(
        (num_groups,),
        group_sizes_data.stride(),
        InfiniDtype.I32,
        device,
        mode="manual",
        set_tensor=group_sizes_data,
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGroupedGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            a.descriptor,
            b.descriptor,
            group_sizes_tensor.descriptor,
        )
    )

    # Invalidate shapes/strides so the kernel can't peek at them through the desc.
    for tensor in [a, b, c, group_sizes_tensor]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGroupedGemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_run():
        check_error(
            LIBINFINIOP.infiniopGroupedGemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                c.data(),
                a.data(),
                b.data(),
                group_sizes_tensor.data(),
                None,  # group_sizes_host: exercise the device-sync fallback path
                ctypes.c_float(alpha),
                ctypes.c_float(beta),
                None,
            )
        )

    lib_run()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c.actual_tensor(), ans_t, atol=atol, rtol=rtol)
    assert torch.allclose(c.actual_tensor(), ans_t, atol=atol, rtol=rtol)

    if PROFILE:
        # fmt: off
        profile_operation("    lib", lambda: lib_run(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyGroupedGemmDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
