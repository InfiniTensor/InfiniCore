import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    TestWorkspace,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration
# ==============================================================================
# Each case: (in_features, out_features, group_size)
_TEST_CASES = [
    (128, 256, 32),
    (512, 2048, 128),
    (1024, 1024, 128),
    # Non-multiple-of-8 edge case for both dims
    (513, 257, 32),
]

_TENSOR_DTYPES = [InfiniDtype.F16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0.0, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# ==============================================================================
#  Reference Implementation (matches CUDA kernel)
# ==============================================================================

def _unpack_qweight_int4_packed_by_rows(qweight_packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """qweight_packed: [in_packed, out_features] int32/uint32.

    Packs 8 input rows per 32-bit word; nibble shift = (row_in_pack * 4).
    Returns: [in_features, out_features] int32 in [0, 15].
    """
    assert qweight_packed.dim() == 2
    shifts = torch.arange(0, 32, 4, device=qweight_packed.device, dtype=torch.int32)
    # [in_packed, 8, out_features]
    expanded = torch.bitwise_right_shift(qweight_packed[:, None, :], shifts[None, :, None])
    vals = torch.bitwise_and(expanded, 0xF).to(torch.int32)
    out_features = qweight_packed.shape[1]
    return vals.reshape(-1, out_features)[:in_features, :]


def _unpack_zeros_int4_packed_by_cols(zeros_packed: torch.Tensor, out_features: int) -> torch.Tensor:
    """zeros_packed: [num_groups, out_packed] int32/uint32.

    Packs 8 output cols per 32-bit word; nibble shift = (col_in_pack * 4).
    Returns: [num_groups, out_features] int32 in [0, 15].
    """
    assert zeros_packed.dim() == 2
    shifts = torch.arange(0, 32, 4, device=zeros_packed.device, dtype=torch.int32)
    # [num_groups, out_packed, 8]
    expanded = torch.bitwise_right_shift(zeros_packed[:, :, None], shifts[None, None, :])
    vals = torch.bitwise_and(expanded, 0xF).to(torch.int32)
    return vals.reshape(zeros_packed.shape[0], -1)[:, :out_features]


def dequantize_gptq_ref(
    qweight_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros_packed: torch.Tensor,
    g_idx: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Reference matches kernel in dequantize_w42f16_nvidia.cu.

    out[row, col] = (q - (z + 1)) * scale
    where q is int4 packed by input rows and z is int4 packed by output cols.
    gid = g_idx[row] (clamped modulo num_groups).
    """
    in_features = g_idx.numel()
    out_features = qweight_packed.shape[1]

    num_groups = scales.shape[0]
    assert scales.shape == (num_groups, out_features)

    q = _unpack_qweight_int4_packed_by_rows(qweight_packed, in_features)  # [in_features, out_features]
    z_full = _unpack_zeros_int4_packed_by_cols(zeros_packed, out_features)  # [num_groups, out_features]

    gid_raw = g_idx.to(torch.int32)
    gid = ((gid_raw % num_groups) + num_groups) % num_groups

    z = z_full[gid]  # [in_features, out_features]
    s = scales[gid]  # [in_features, out_features]

    out = (q - (z + 1)).to(torch.float32) * s.to(torch.float32)
    return out.to(torch.float16)


def _make_packed_inputs(in_features: int, out_features: int, group_size: int, torch_device: str):
    # num_groups is implicit (same convention as GPTQ: group per input channel)
    num_groups = (in_features + group_size - 1) // group_size
    in_packed = (in_features + 7) // 8
    out_packed = (out_features + 7) // 8

    # Deterministic group mapping
    g_idx = (torch.arange(in_features, device=torch_device, dtype=torch.int32) // group_size)

    # Random nibble-level values to avoid relying on sign/shift corner cases
    q_nib = torch.randint(0, 16, (in_features, out_features), device=torch_device, dtype=torch.int32)
    z_nib = torch.randint(0, 16, (num_groups, out_features), device=torch_device, dtype=torch.int32)

    # scales in fp16
    scales = (torch.rand((num_groups, out_features), device=torch_device, dtype=torch.float16) * 0.5 + 0.01)

    # Pack qweight: [in_packed, out_features]
    qweight_packed = torch.zeros((in_packed, out_features), device=torch_device, dtype=torch.int32)
    for i in range(8):
        rows = torch.arange(i, in_features, 8, device=torch_device)
        if rows.numel() == 0:
            continue
        pack_rows = (rows // 8).to(torch.int64)
        qweight_packed[pack_rows, :] |= (q_nib[rows, :] & 0xF) << (i * 4)

    # Pack zeros: [num_groups, out_packed]
    zeros_packed = torch.zeros((num_groups, out_packed), device=torch_device, dtype=torch.int32)
    for j in range(8):
        cols = torch.arange(j, out_features, 8, device=torch_device)
        if cols.numel() == 0:
            continue
        pack_cols = (cols // 8).to(torch.int64)
        zeros_packed[:, pack_cols] |= (z_nib[:, cols] & 0xF) << (j * 4)

    return qweight_packed, scales, zeros_packed, g_idx


# ==============================================================================
#  Test Entrypoint
# ==============================================================================

def test(
    handle,
    device,
    in_features,
    out_features,
    group_size,
    dtype=None,
    sync=None,
):
    print(
        f"Testing Dequantize GPTQ on {InfiniDeviceNames[device]} with in_features:{in_features}, out_features:{out_features}, group_size:{group_size}"
    )

    # Infer torch device from a probe tensor created by TestTensor.
    probe = TestTensor((1,), None, InfiniDtype.U8, device, mode="ones")
    torch_device = probe.actual_tensor().device

    qweight_packed, scales, zeros_packed, g_idx = _make_packed_inputs(
        in_features, out_features, group_size, torch_device
    )

    # Reference
    ans = dequantize_gptq_ref(qweight_packed, scales, zeros_packed, g_idx, group_size)

    # Wrap into TestTensor so we get descriptors + raw pointers
    qweight = TestTensor.from_torch(qweight_packed, InfiniDtype.I32, device)
    zeros = TestTensor.from_torch(zeros_packed, InfiniDtype.I32, device)
    qscales = TestTensor.from_torch(scales, InfiniDtype.F16, device)
    g_idx_t = TestTensor.from_torch(g_idx, InfiniDtype.I32, device)

    out = TestTensor((in_features, out_features), None, InfiniDtype.F16, device, mode="zeros")

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDequantizeGPTQDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            qweight.descriptor,
            qscales.descriptor,
            zeros.descriptor,
            g_idx_t.descriptor,
        )
    )

    # Invalidate descriptors (same pattern as other tests)
    for tensor in [qweight, zeros, qscales, g_idx_t, out]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetDequantizeGPTQWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_dequantize_gptq():
        check_error(
            LIBINFINIOP.infiniopDequantizeGPTQ(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                qweight.data(),
                qscales.data(),
                zeros.data(),
                g_idx_t.data(),
                None,
            )
        )

    lib_dequantize_gptq()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation("PyTorch", lambda: dequantize_gptq_ref(qweight_packed, scales, zeros_packed, g_idx, group_size), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_dequantize_gptq(), device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyDequantizeGPTQDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # This operator is intended to run on NVIDIA in our current environment.
    # Make `srun ... python test/infiniop/dequantize_gptq.py` work without extra flags.
    if not any(
        getattr(args, name)
        for name in [
            "cpu",
            "nvidia",
            "iluvatar",
            "qy",
            "cambricon",
            "ascend",
            "metax",
            "moore",
            "kunlun",
            "hygon",
        ]
    ):
        args.nvidia = True

    devices = [d for d in get_test_devices(args) if InfiniDeviceNames[d] == "NVIDIA"]
    if not devices:
        raise RuntimeError("No NVIDIA device selected; run with --nvidia under srun.")

    for device in devices:
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
