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
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules

# 使用真实的图像参数进行测试
H, W, D_PATCH = 336, 476, 14
HP = H // D_PATCH  # 24
WP = W // D_PATCH  # 34

# 根据 pos_ids.rs 的实现，计算实际的序列长度
# pos_ids.rs: ptr 从 0 开始，按 2x2 块遍历
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

_TEST_CASES_ = [
    # (shape, x_strides, y_strides) - 使用实际计算的序列长度
    ((ACTUAL_SEQ_LEN, 32, 128), None, None),  # 2D MRoPE: dhead = table_dim * 4, so 128 = 32 * 4
    ((ACTUAL_SEQ_LEN, 16, 64), None, None),   # 64 = 16 * 4
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-3},
}


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_X,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


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


def multimodal_rotary_embedding_2d(ans, t, pos_ids, sin, cos, device):
    """
    2D MRoPE implementation for reference
    pos_ids shape: [seq_len, 2] - (h, w) positions
    sin/cos shape: [max_pos, dh//4] - table for each dimension
    """
    seq_len, n_head, dh = t.shape
    dt = t.dtype
    assert dh % 4 == 0, "Embedding dimension must be divisible by 4 for 2D MRoPE."

    dh_div_4 = dh // 4
    dh_div_2 = dh // 2

    if device == InfiniDeviceEnum.CPU:
        t = t.float()
        sin = sin.float()
        cos = cos.float()

    # Apply rotation based on your .cuh implementation
    for seq_idx in range(seq_len):
        for head_idx in range(n_head):
            for i in range(dh_div_2):
                # 2 维 mrope 的 w, h 维度均分 d_div_2，每个分到 d_div_2 / 2
                id_h = i // (dh_div_2 // 2)  # w, h 的维度索引
                id_l = i % (dh_div_2 // 2)   # w, h 维度内索引
                pos = pos_ids[seq_idx, id_h].item()  # 2 维 pos 的 shape: [seq_len, 2], strides: [2, 1]

                sin_val = sin[pos, id_l].item()
                cos_val = cos[pos, id_l].item()

                # Apply rotation
                a = t[seq_idx, head_idx, i].item()
                b = t[seq_idx, head_idx, i + dh_div_2].item()

                ans[seq_idx, head_idx, i] = (a * cos_val - b * sin_val)
                ans[seq_idx, head_idx, i + dh_div_2] = (a * sin_val + b * cos_val)

    if device == InfiniDeviceEnum.CPU:
        ans = ans.to(dt)


def sin_cos_table_2d(max_pos, dim, device, theta, dtype):
    """Generate sin/cos table for 2D MRoPE"""
    assert dim % 4 == 0, "Embedding dimension must be divisible by 4 for 2D MRoPE."
    dh_div_4 = dim // 4

    # Create frequency for each dimension component
    freqs = 1.0 / (theta ** (torch.arange(0, dh_div_4, 1).float() / dh_div_4))
    pos = torch.arange(0, max_pos, dtype=torch.float32)
    angles = torch.outer(pos, freqs)

    return (
        TestTensor.from_torch(torch.sin(angles), dtype, device),
        TestTensor.from_torch(torch.cos(angles), dtype, device),
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
    seq_len, n_head, dh = shape

    # For 2D MRoPE, dh must be divisible by 4
    if dh % 4 != 0:
        return

    print(
        f"Testing 2D MRoPE on {InfiniDeviceNames[device]} with shape:{shape} x_strides:{x_strides} y_strides:{y_strides} and dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
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
    assert pos_ids.shape[0] == seq_len, f"pos_ids length {pos_ids.shape[0]} != seq_len {seq_len}"

    max_pos = pos_ids.torch_tensor().max().item() + 1

    # Generate sin/cos tables
    sin_table, cos_table = sin_cos_table_2d(max_pos, dh, device, 10000.0, dtype)

    # Compute reference result
    multimodal_rotary_embedding_2d(
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
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: multimodal_rotary_embedding_2d(
                y.torch_tensor(),
                x.torch_tensor(),
                pos_ids.torch_tensor(),
                sin_table.torch_tensor(),
                cos_table.torch_tensor(),
                device,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "InfiniOP",
            lib_mrope2d,
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )

    check_error(LIBINFINIOP.infiniopDestroyMRoPE2DDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    print(f"2D MRoPE Test Configuration:")
    print(f"  Image size: {H}x{W}, Patch size: {D_PATCH}")
    print(f"  Calculated sequence length: {ACTUAL_SEQ_LEN}")

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92m2D MRoPE Test passed!\033[0m")
