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
_TEST_CASES_ = [
    # (shape, x_strides, y_strides)
    ((1, 32, 128), None, None),
    ((10, 32, 64), None, None),
    # 昇腾暂不满足这个用例，最后一维度 <=32 会有问题，可能与其核心
    # 接口 GatherMask 的内部实现相关，目前 48 64 128 都可以支持
    ((4, 1, 32), (64, 64, 1), None),
    ((11, 33, 128), None, (8000, 200, 1)),
    ((3, 32, 128), (8000, 200, 1), (7000, 128, 1)),
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


class Algorithm(Enum):
    GPT_J = 0
    GPT_NEOX = 1


_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_X,
]


_ALGO = [
    Algorithm.GPT_J,
    Algorithm.GPT_NEOX,
]

_TEST_CASES = [
    test_case + (inplace_item, algo_item)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
    for algo_item in _ALGO
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def rotary_embedding(ans, t, sin, cos, device, algo):
    def _torch_rope(sin, cos, t1, t2):
        cos = cos.unsqueeze(1)  # [seq_len, 1, dh // 2]
        sin = sin.unsqueeze(1)  # [seq_len, 1, dh // 2]
        if device == InfiniDeviceEnum.CPU:
            (t1, t2, cos, sin) = (
                t1.float(),
                t2.float(),
                cos.float(),
                sin.float(),
            )

        t_out_1 = t1 * cos - t2 * sin
        t_out_2 = t1 * sin + t2 * cos

        return t_out_1, t_out_2


    dh = t.shape[-1]
    dt = t.dtype
    assert dh % 2 == 0, "Embedding dimension must be even."

    if algo == Algorithm.GPT_J:
        t_even = t[..., 0::2]  # [seq_len, n_head, dh // 2]
        t_odd = t[..., 1::2]  # [seq_len, n_head, dh // 2]

        t_out_even, t_out_odd = _torch_rope(sin, cos, t_even, t_odd)

        ans[..., 0::2] = t_out_even.to(dt)
        ans[..., 1::2] = t_out_odd.to(dt)
    else:
        half_dim = dh // 2   
        t_first = t[..., :half_dim]
        t_second = t[..., half_dim:]

        t_out_first, t_out_second = _torch_rope(sin, cos, t_first, t_second)

        ans[..., :half_dim] = t_out_first.to(dt)
        ans[..., half_dim:] = t_out_second.to(dt)


def sin_cos_table(pos, dim, device, theta, dtype):
    assert dim % 2 == 0, "Embedding dimension must be even."
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    angles = torch.outer(pos.cpu(), freqs)
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
    algo=Algorithm.GPT_J,
    dtype=torch.float32,
    sync=None,
):
    x = TestTensor(shape, x_strides, dtype, device)
    if inplace == Inplace.INPLACE_X:
        if x_strides != y_strides:
            return
        y = x
    else:
        y = TestTensor(shape, y_strides, dtype, device)

    print(
        f"Testing Rotary Positional Embedding on {InfiniDeviceNames[device]} with shape:{shape} x_strides:{x_strides} y_strides:{y_strides} and dtype:{InfiniDtypeNames[dtype]} inplace:{inplace} algo:{algo}"
    )
    theta = 1e5
    pos = TestTensor.from_torch(torch.arange(0, x.shape[0]), InfiniDtype.I32, device)
    sin_table, cos_table = sin_cos_table(
        pos.torch_tensor(), x.shape[2], x.device, theta, dtype
    )

    rotary_embedding(
        y.torch_tensor(),
        x.torch_tensor(),
        sin_table.torch_tensor(),
        cos_table.torch_tensor(),
        device,
        algo,
    )

    descriptor = infiniopOperatorDescriptor_t()

    if sync is not None:
        sync()

    check_error(
        LIBINFINIOP.infiniopCreateRoPEDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            pos.descriptor,
            sin_table.descriptor,
            cos_table.descriptor,
            algo.value,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [y, x, pos, sin_table, cos_table]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRoPEWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_rope():
        check_error(
            LIBINFINIOP.infiniopRoPE(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                pos.data(),
                sin_table.data(),
                cos_table.data(),
                None,
            )
        )

    lib_rope()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: rotary_embedding(
                y.torch_tensor(),
                x.torch_tensor(),
                sin_table.torch_tensor(),
                cos_table.torch_tensor(),
                device,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_rope(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyRoPEDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
