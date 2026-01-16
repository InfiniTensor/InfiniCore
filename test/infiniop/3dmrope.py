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
#  Configuration based on Qwen2-VL for 3D MRoPE
# ==============================================================================

# 3D parameters for video/temporal modeling
T_DIM = 3
H_IMG = 224
W_IMG = 224
D_PATCH = 14
PRE_TEXT_LEN = 4
POST_TEXT_LEN = 5


def calculate_3d_seq_len(t, h, w, d_patch, pre_text_len, post_text_len):
    """Calculate sequence length according to 3D pos_ids.rs implementation"""
    spatial_merge_size = 2
    t_len = t
    h_len = h // d_patch // spatial_merge_size
    w_len = w // d_patch // spatial_merge_size
    vision_len = t_len * h_len * w_len
    total_len = pre_text_len + vision_len + post_text_len
    return total_len


ACTUAL_SEQ_LEN = calculate_3d_seq_len(
    T_DIM, H_IMG, W_IMG, D_PATCH, PRE_TEXT_LEN, POST_TEXT_LEN)

# 注意：根据 Qwen2-VL，形状是 [nhead, seqlen, dhead]
# 3D MRoPE: dhead = table_dim * 2
_TEST_CASES_ = [
    # (shape, x_strides, y_strides) - 形状：[nhead, seqlen, dhead]
    # 3D MRoPE: dhead = table_dim * 2
    # ((32, ACTUAL_SEQ_LEN, 128), None, None),   # 大规模测试: 32头, 128维
    # ((16, ACTUAL_SEQ_LEN, 64), None, None),    # 中等规模: 16头, 64维
    ((8, ACTUAL_SEQ_LEN, 32), None, None),     # 小规模: 8头, 32维
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types (stricter for testing optimization)
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 5e-4, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 5e-4, "rtol": 1e-5},
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


def generate_3d_pos_ids(t, h, w, d_patch, pre_text_len, post_text_len, device, dtype=InfiniDtype.I32):
    """Generate 3D position IDs according to pos_ids.rs implementation"""
    spatial_merge_size = 2
    t_len = t
    h_len = h // d_patch // spatial_merge_size
    w_len = w // d_patch // spatial_merge_size
    vision_len = t_len * h_len * w_len
    total_len = pre_text_len + vision_len + post_text_len

    pos = []
    idx = 0

    # 图像前文本
    for i in range(pre_text_len):
        pos.append([i, i, i])
        idx += 1

    # 图像
    img_start_pos = pre_text_len
    for t_idx in range(t_len):
        for h_idx in range(h_len):
            for w_idx in range(w_len):
                t_pos = img_start_pos + t_idx
                h_pos = img_start_pos + h_idx
                w_pos = img_start_pos + w_idx
                pos.append([t_pos, h_pos, w_pos])
                idx += 1

    # 图像后文本
    t_max_pos = img_start_pos + t_len - 1
    h_max_pos = img_start_pos + h_len - 1
    w_max_pos = img_start_pos + w_len - 1
    image_max_pos = max(t_max_pos, h_max_pos, w_max_pos)
    text_start_pos = image_max_pos + 1
    for i in range(post_text_len):
        pos_val = text_start_pos + i
        pos.append([pos_val, pos_val, pos_val])
        idx += 1

    assert idx == total_len
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


def multimodal_rotary_embedding_3d_reference(ans, t, pos_ids, sin_table, cos_table, rope_section, device):
    """
    3D MRoPE reference implementation based on Qwen2-VL style
    t: [nhead, seqlen, dhead]
    pos_ids: [seqlen, 3] - (t, h, w) positions
    sin_table/cos_table: [max_pos, dhead//2] - table for all dimensions
    rope_section: [3] - section boundaries for t, h, w dimensions
    """
    nhead, seqlen, dhead = t.shape
    dt = t.dtype
    assert dhead % 2 == 0, "Embedding dimension must be divisible by 2 for 3D MRoPE."

    dhead_div_2 = dhead // 2

    if device == InfiniDeviceEnum.CPU:
        t = t.float()
        sin_table = sin_table.float()
        cos_table = cos_table.float()

    # Process each sequence position and head directly (matching CUDA kernel logic)
    for seq_idx in range(seqlen):
        for head_idx in range(nhead):
            for i in range(dhead_div_2):
                # Find i in rope_section (matching CUDA kernel logic)
                thw = 0
                for j in range(3):
                    if i < rope_section[j].item():
                        thw = j
                        break

                # Get position index
                pos = pos_ids[seq_idx, thw].item()

                sin_val = sin_table[pos, i].item()
                cos_val = cos_table[pos, i].item()

                a = t[head_idx, seq_idx, i].item()
                b = t[head_idx, seq_idx, i + dhead_div_2].item()

                ans[head_idx, seq_idx, i] = a * cos_val - b * sin_val
                ans[head_idx, seq_idx, i + dhead_div_2] = a * \
                    sin_val + b * cos_val


def sin_cos_table_3d(max_pos, dim, device, theta, dtype):
    """Generate sin/cos table for 3D MRoPE"""
    assert dim % 2 == 0, "Embedding dimension must be divisible by 2 for 3D MRoPE."
    dh_div_2 = dim // 2

    # Create frequency for all dimensions (unified table)
    freqs = 1.0 / (theta ** (torch.arange(0, dh_div_2, 1).float() / dh_div_2))
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

    # For 3D MRoPE, dh must be divisible by 2
    if dhead % 2 != 0:
        return

    print(
        f"Testing 3D MRoPE (Qwen2-VL style) on {InfiniDeviceNames[device]} with shape:[{nhead}, {seqlen}, {dhead}] dtype:{InfiniDtypeNames[dtype]}"
    )

    x = TestTensor(shape, x_strides, dtype, device)
    if inplace == Inplace.INPLACE_X:
        if x_strides != y_strides:
            return
        y = x
    else:
        y = TestTensor(shape, y_strides, dtype, device)

    # Generate 3D position IDs using real parameters
    t_dim, h_dim, w_dim, d_patch = T_DIM, H_IMG, W_IMG, D_PATCH
    pre_text_len, post_text_len = PRE_TEXT_LEN, POST_TEXT_LEN

    pos_ids = generate_3d_pos_ids(
        t_dim, h_dim, w_dim, d_patch, pre_text_len, post_text_len, device)

    # Verify the sequence length matches
    assert pos_ids.shape[0] == seqlen, f"pos_ids length {pos_ids.shape[0]} != seqlen {seqlen}"

    # Calculate max_pos_val for sin/cos table
    spatial_merge_size = 2
    t_len = t_dim
    h_len = h_dim // d_patch // spatial_merge_size
    w_len = w_dim // d_patch // spatial_merge_size
    img_start_pos = pre_text_len
    t_max_pos = img_start_pos + t_len - 1
    h_max_pos = img_start_pos + h_len - 1
    w_max_pos = img_start_pos + w_len - 1
    image_max_pos = max(t_max_pos, h_max_pos, w_max_pos)
    text_start_pos = image_max_pos + 1
    max_pos_val = text_start_pos + post_text_len

    # Generate sin/cos tables
    sin_table, cos_table = sin_cos_table_3d(
        max_pos_val, dhead, device, 10000.0, dtype)

    # rope_section represents the accumulated dimensions for t, h, w
    # For example, if dhead=32, we might split as [10, 11, 11] -> accumulated [10, 21, 32]
    dhead_div_2 = dhead // 2
    section_t = dhead_div_2 // 3
    section_h = dhead_div_2 // 3
    section_w = dhead_div_2 - section_t - section_h  # remainder
    rope_section = TestTensor.from_torch(
        torch.tensor([section_t, section_t + section_h,
                     dhead_div_2], dtype=torch.int32),
        InfiniDtype.I32,
        device
    )

    # Compute reference result using Qwen2-VL style
    multimodal_rotary_embedding_3d_reference(
        y.torch_tensor(),
        x.torch_tensor(),
        pos_ids.torch_tensor(),
        sin_table.torch_tensor(),
        cos_table.torch_tensor(),
        rope_section.torch_tensor(),
        device,
    )

    # Create descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateMRoPE3DDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            pos_ids.descriptor,
            sin_table.descriptor,
            cos_table.descriptor,
            rope_section.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [y, x, pos_ids, sin_table, cos_table, rope_section]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetMRoPE3DWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_mrope3d():
        check_error(
            LIBINFINIOP.infiniopMRoPE3D(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                pos_ids.data(),
                sin_table.data(),
                cos_table.data(),
                rope_section.data(),
                None,
            )
        )

    lib_mrope3d()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        print(f"Expected shape: {y.torch_tensor().shape}")
        print(f"Actual shape: {y.actual_tensor().shape}")
        print(f"pos_ids shape: {pos_ids.torch_tensor().shape}")
        print(f"sin_table shape: {sin_table.torch_tensor().shape}")
        print(f"rope_section shape: {rope_section.torch_tensor().shape}")
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

    check_error(LIBINFINIOP.infiniopDestroyMRoPE3DDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    print(f"3D MRoPE Test Configuration (Qwen2-VL style):")
    print(
        f"  Video/Image size: {T_DIM}x{H_IMG}x{W_IMG}, Patch size: {D_PATCH}")
    print(
        f"  Pre-text length: {PRE_TEXT_LEN}, Post-text length: {POST_TEXT_LEN}")
    print(f"  Calculated sequence length: {ACTUAL_SEQ_LEN}")
    print(f"  Tensor shape format: [nhead, seqlen, dhead]")

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92m3D MRoPE Test (Qwen2-VL style) passed!\033[0m")
