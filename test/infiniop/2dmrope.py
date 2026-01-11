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
    InfiniDeviceEnum,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto

# ==============================================================================
#  Configuration based on Qwen2-VL
# ==============================================================================

# 使用真实的图像参数进行测试
H, W, D_PATCH = 336, 476, 14
HP = H // D_PATCH  # 24
WP = W // D_PATCH  # 34

# 根据 pos_ids.rs 的实现，计算实际的序列长度


def calculate_2d_seq_len(h, w, d_patch):
    hp = h // d_patch
    wp = w // d_patch
    count = 0
    for y in range(0, hp, 2):
        for x in range(0, wp, 2):
            for dy in range(2):
                for dx in range(2):
                    if y + dy < hp and x + dx < wp:
                        count += 1
    return count


ACTUAL_SEQ_LEN = calculate_2d_seq_len(H, W, D_PATCH)

# 注意：根据 Qwen2-VL，形状是 [nhead, seqlen, dhead]
_TEST_CASES_ = [
    # (shape, x_strides, y_strides) - 形状：[nhead, seqlen, dhead]
    # 2D MRoPE: dhead = table_dim * 4
    ((32, ACTUAL_SEQ_LEN, 128), None, None),   # 大规模测试: 32头, 128维
    ((16, ACTUAL_SEQ_LEN, 64), None, None),    # 中等规模: 16头, 64维
    ((8, ACTUAL_SEQ_LEN, 32), None, None),     # 小规模: 8头, 32维
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16]
# InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-4},
    InfiniDtype.BF16: {"atol": 8e-3, "rtol": 1e-4},
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-9},
}


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


_INPLACE = [
    Inplace.OUT_OF_PLACE,
    # Inplace.INPLACE_X,  # 先测试非原地操作
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 1
NUM_ITERATIONS = 3


def generate_2d_pos_ids(h, w, d_patch, device, dtype=InfiniDtype.I32):
    """Generate 2D position IDs according to pos_ids.rs implementation"""
    hp = h // d_patch
    wp = w // d_patch
    pos = []

    # Following the Rust implementation exactly
    for y in range(0, hp, 2):
        for x in range(0, wp, 2):
            for dy in range(2):
                for dx in range(2):
                    if y + dy < hp and x + dx < wp:
                        pos.append([y + dy, x + dx])

    pos_ids = torch.tensor(pos, dtype=torch.int32)
    return TestTensor.from_torch(pos_ids, dtype, device)


def rotate_half(x):
    """Rotates half the hidden dims of the input. (Qwen2-VL style)"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision_reference(tensor, freqs):
    """
    Reference implementation based on Qwen2-VL's apply_rotary_pos_emb_vision
    tensor: [nhead, seqlen, dhead]
    freqs: [seqlen, dhead//2] or similar
    """
    orig_dtype = tensor.dtype
    tensor = tensor.float()

    # freqs should contain the angles for rotation
    cos = freqs.cos()  # [seqlen, dhead//2]
    sin = freqs.sin()  # [seqlen, dhead//2]

    # Expand to match tensor dimensions
    # cos/sin: [seqlen, dhead//2] -> [seqlen, dhead]
    cos = cos.repeat(1, 2)  # [seqlen, dhead]
    sin = sin.repeat(1, 2)  # [seqlen, dhead]

    # Add batch dimension for broadcasting with [nhead, seqlen, dhead]
    cos = cos.unsqueeze(0)  # [1, seqlen, dhead]
    sin = sin.unsqueeze(0)  # [1, seqlen, dhead]

    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


def multimodal_rotary_embedding_2d_reference(ans, t, pos_ids, sin_table, cos_table, device):
    """
    2D MRoPE reference implementation based on Qwen2-VL style
    t: [nhead, seqlen, dhead]
    pos_ids: [seqlen, 2] - (h, w) positions
    sin_table/cos_table: [max_pos, dhead//4] - table for each dimension
    """
    nhead, seqlen, dhead = t.shape
    dt = t.dtype
    assert dhead % 4 == 0, "Embedding dimension must be divisible by 4 for 2D MRoPE."

    dhead_div_2 = dhead // 2
    dhead_div_4 = dhead // 4

    if device == InfiniDeviceEnum.CPU:
        t = t.float()
        sin_table = sin_table.float()
        cos_table = cos_table.float()

    # Create frequency tensor for each position
    # This mimics the freqs parameter in apply_rotary_pos_emb_vision
    freqs = torch.zeros(seqlen, dhead_div_2,
                        dtype=torch.float32, device=t.device)

    for seq_idx in range(seqlen):
        for i in range(dhead_div_2):
            # 2 维 mrope 的 w, h 维度均分 dhead_div_2，每个分到 dhead_div_4
            dim_idx = i // dhead_div_4  # 0 for h, 1 for w
            within_dim_idx = i % dhead_div_4  # index within dimension

            pos = pos_ids[seq_idx, dim_idx].item()
            freqs[seq_idx, i] = torch.atan2(
                sin_table[pos, within_dim_idx],
                cos_table[pos, within_dim_idx]
            )

    # Apply rotary embedding using Qwen2-VL style
    ans[:] = apply_rotary_pos_emb_vision_reference(t, freqs)


def sin_cos_table_2d(max_pos, dim, device, theta, dtype):
    """Generate sin/cos table for 2D MRoPE"""
    assert dim % 4 == 0, "Embedding dimension must be divisible by 4 for 2D MRoPE."
    dh_div_4 = dim // 4

    # Create frequency for each dimension component
    freqs = 1.0 / (theta ** (torch.arange(0, dh_div_4, 1).float() / dh_div_4))
    pos = torch.arange(0, max_pos, dtype=torch.float32)
    angles = torch.outer(pos, freqs)

    return (
        TestTensor.from_torch(torch.sin(angles), InfiniDtype.F32, device),
        TestTensor.from_torch(torch.cos(angles), InfiniDtype.F32, device),
    )


def test(
    handle,
    device,
    shape,
    x_strides=None,
    y_strides=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float32,
    sync=None,
):
    nhead, seqlen, dhead = shape

    # For 2D MRoPE, dh must be divisible by 4
    if dhead % 4 != 0:
        return

    print(
        f"Testing 2D MRoPE (Qwen2-VL style) on {InfiniDeviceNames[device]} with shape:[{nhead}, {seqlen}, {dhead}] dtype:{InfiniDtypeNames[dtype]}"
    )

    x = TestTensor(shape, x_strides, dtype, device)
    if inplace == Inplace.INPLACE_X:
        if x_strides != y_strides:
            return
        y = x
    else:
        y = TestTensor(shape, y_strides, dtype, device)

    # Generate 2D position IDs using real parameters
    h, w, d_patch = H, W, D_PATCH
    pos_ids = generate_2d_pos_ids(h, w, d_patch, device)

    # Verify the sequence length matches
    assert pos_ids.shape[0] == seqlen, f"pos_ids length {pos_ids.shape[0]} != seqlen {seqlen}"

    max_pos = pos_ids.torch_tensor().max().item() + 1

    # Generate sin/cos tables
    sin_table, cos_table = sin_cos_table_2d(
        max_pos, dhead, device, 10000.0, dtype)

    # Compute reference result using Qwen2-VL style
    multimodal_rotary_embedding_2d_reference(
        y.torch_tensor(),
        x.torch_tensor(),
        pos_ids.torch_tensor(),
        sin_table.torch_tensor(),
        cos_table.torch_tensor(),
        device,
    )

    # Create descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateMRoPE2DDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            pos_ids.descriptor,
            sin_table.descriptor,
            cos_table.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [y, x, pos_ids, sin_table, cos_table]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetMRoPE2DWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_mrope2d():
        check_error(
            LIBINFINIOP.infiniopMRoPE2D(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                pos_ids.data(),
                sin_table.data(),
                cos_table.data(),
                None,
            )
        )

    lib_mrope2d()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        print(f"Expected shape: {y.torch_tensor().shape}")
        print(f"Actual shape: {y.actual_tensor().shape}")
        print(f"pos_ids shape: {pos_ids.torch_tensor().shape}")
        print(f"sin_table shape: {sin_table.torch_tensor().shape}")
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    success = torch.allclose(
        y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    if success:
        print("✅ Test PASSED!")
    else:
        print("❌ Test FAILED!")
        if not DEBUG:
            print("Run with --debug to see detailed comparison")
            # Show a brief comparison
            diff = torch.abs(y.actual_tensor() - y.torch_tensor())
            print(f"Max absolute difference: {diff.max().item():.6f}")
            print(f"Mean absolute difference: {diff.mean().item():.6f}")

    assert success

    check_error(LIBINFINIOP.infiniopDestroyMRoPE2DDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    print(f"2D MRoPE Test Configuration (Qwen2-VL style):")
    print(f"  Image size: {H}x{W}, Patch size: {D_PATCH}")
    print(f"  Calculated sequence length: {ACTUAL_SEQ_LEN}")
    print(f"  Tensor shape format: [nhead, seqlen, dhead]")

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92m2D MRoPE Test (Qwen2-VL style) passed!\033[0m")
