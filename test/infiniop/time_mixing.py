import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    create_workspace
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, y_stride, r_stride, w_stride, k_stride, v_stride, a_stride, b_stride
    # ((1, 2, 128), None, None, None, None, None, None, None),
    ((1, 3, 768), None, None, None, None, None, None, None),
]

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
# _TENSOR_DTYPES = [torch.float16, torch.float32]
# _TENSOR_DTYPES = [torch.float32]
_TENSOR_DTYPES = [torch.float16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-2},
    torch.float32: {"atol": 1e-4, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 200
HEAD_SIZE = 64

class TimeMixingDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopTimeMixingDescriptor_t = POINTER(TimeMixingDescriptor)


def time_mixing(r, w, k, v, a, b, dtype):
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)

    return out.view(B, T, C).to(dtype=dtype)


def test(
    lib,
    handle,
    torch_device,
    shape,
    y_stride=None,
    r_stride=None,
    w_stride=None,
    k_stride=None,
    v_stride=None,
    a_stride=None,
    b_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    print(
        f"Testing TimeMixing on {torch_device} with shape:{shape}"
        f"dtype:{dtype} inplace:{inplace}"
    )
    y = torch.rand(shape, dtype=dtype).to(torch_device)
    r = torch.rand(shape, dtype=dtype).to(torch_device)
    w = torch.rand(shape, dtype=dtype).to(torch_device)
    k = torch.rand(shape, dtype=dtype).to(torch_device)
    v = torch.rand(shape, dtype=dtype).to(torch_device)
    a = torch.rand(shape, dtype=dtype).to(torch_device)
    b = torch.rand(shape, dtype=dtype).to(torch_device)

    ans = time_mixing(r, w, k, v, a, b, dtype)

    r_tensor, w_tensor, k_tensor, v_tensor, a_tensor, b_tensor = [to_tensor(tensor, lib) for tensor in [r, w, k, v, a, b]]
    y_tensor = to_tensor(y, lib)
    if sync is not None:
        sync()

    descriptor = infiniopTimeMixingDescriptor_t()
    check_error(
        lib.infiniopCreateTimeMixingDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            r_tensor.descriptor,
            w_tensor.descriptor,
            k_tensor.descriptor,
            v_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [y_tensor, r_tensor, w_tensor, k_tensor, v_tensor, a_tensor, b_tensor]:
        tensor.destroyDesc(lib)

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetTimeMixingWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, y.device)

    def lib_time_mixing():
        check_error(
            lib.infiniopTimeMixing(
                descriptor, 
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                y_tensor.data, r_tensor.data, w_tensor.data,
                k_tensor.data, v_tensor.data, a_tensor.data, b_tensor.data, None
            )
        )

    lib_time_mixing()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y, ans, atol=atol, rtol=rtol)
    assert torch.allclose(y, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: time_mixing(r, w, k, v, a, b, dtype), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_time_mixing(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroyTimeMixingDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateTimeMixingDescriptor.restype = c_int32
    lib.infiniopCreateTimeMixingDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopTimeMixingDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetTimeMixingWorkspaceSize.restype = c_int32
    lib.infiniopGetTimeMixingWorkspaceSize.argtypes = [
        infiniopTimeMixingDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopTimeMixing.restype = c_int32
    lib.infiniopTimeMixing.argtypes = [
        infiniopTimeMixingDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyTimeMixingDescriptor.restype = c_int32
    lib.infiniopDestroyTimeMixingDescriptor.argtypes = [
        infiniopTimeMixingDescriptor_t,
    ]

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
