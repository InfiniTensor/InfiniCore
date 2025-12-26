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
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # x_shape, w_shape, symmetric, bias_exit, y_shape
    ((128, 512), (512, 1024), True, False, (128, 1024)),
    ((128, 128), (128, 128), False, True, (128, 128)),
    ((256, 1024), (1024, 2048), True, False, (256, 2048)),
    ((256, 2048), (2048, 1024), False, True, (256, 1024)),
    ((1024, 2048), (2048, 4096), True, False, (1024, 4096)),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 5e-2},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 3e-5, "rtol": 5e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def linearFunction(bias, x, w):
    if bias is not None:
        ans = (
           torch.matmul(x.to(torch.float32), w.to(torch.float32)).to(x.dtype)
            + bias
        )
    else:
        ans = (
            torch.matmul(x.to(torch.float32), w.to(torch.float32)).to(x.dtype)
        )
    return ans
def per_token_quant_int8_torch(x, symmetric):
    if symmetric:
        x = x.float()
        absmax = x.abs().max(dim=-1).values
        absmax = absmax.clamp_min(1e-10).unsqueeze(-1)
        scale_x = absmax / 127
        x_q = x.mul(127 / absmax)
        x_q = torch.round(x_q).to(torch.int8)

        return x_q, scale_x, None
    else:
        w = x.float()
        w_min = w.min(dim=-1, keepdim=True)[0]
        w_max = w.max(dim=-1, keepdim=True)[0]

        # 避免除以零
        w_scale = (w_max - w_min) / 255.0
        w_scale = torch.clamp(w_scale, min=1e-8)

        # 计算zero point
        w_zero = -w_min / w_scale - 128.0

        # 计算量化值
        w_q = torch.round(w / w_scale + w_zero)

        # 限制范围[-128, 127]
        w_q = torch.clamp(w_q, -128, 127)

        # 转为int8
        w_packed = w_q.to(torch.int8)

        return w_packed, w_scale, w_zero
    
def computeQuant(
        handle,
        device,
        x, 
        symmetric,
        sync=None,
):
    x_shape = x.shape
    M, K = x_shape

    x_packed = TestTensor(x_shape, None, InfiniDtype.I8, device, mode="zeros")
    x_scale = TestTensor((M, 1), None, InfiniDtype.F32, device)
    if symmetric:
        x_zero = None
    else:
        x_zero = TestTensor((M, 1), None, InfiniDtype.F32, device)
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreatePerChannelQuantI8Descriptor(
            handle,
            ctypes.byref(descriptor),
            x_packed.descriptor,
            x_scale.descriptor,
            None if symmetric else x_zero.descriptor,
            x.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()
    x_packed.destroy_desc()
    x_scale.destroy_desc()
    if symmetric == False:
        x_zero.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetPerChannelQuantI8WorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)
    
    def lib_per_channel_quant_int8():
        check_error(
            LIBINFINIOP.infiniopPerChannelQuantI8(
                descriptor,
                workspace.data(),
                workspace_size.value,
                x_packed.data(),
                x_scale.data(),
                None if symmetric else x_zero.data(),
                x.data(),
                None,
            )
        )

    lib_per_channel_quant_int8()
    
    if sync is not None:
        sync()
    check_error(LIBINFINIOP.infiniopDestroyPerChannelQuantI8Descriptor(descriptor))
    if symmetric:
        return x_packed.actual_tensor(), x_scale.actual_tensor(), None
    else:
        return x_packed.actual_tensor(), x_scale.actual_tensor(), x_zero.actual_tensor()

def test(
    handle,
    device,
    x_shape,
    w_shape,
    symmetric,
    bias_exit,
    y_shape,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F16,
    sync=None,
):
    if symmetric == False:
        return
    print(
        f"Testing W8A8 on {InfiniDeviceNames[device]} with x_shape:{x_shape}, w_shape:{w_shape}, symmetric:{symmetric}, bias:{bias_exit}, inplace:{inplace} dtype:{InfiniDtypeNames[dtype]}"
    )
    M, K = x_shape
    N = w_shape[1]
    if bias_exit:
        bias = TestTensor((N,), None, dtype, device)
    else:
        bias = None
    x = TestTensor(x_shape, None, dtype, device)
    w = TestTensor(w_shape, None, dtype, device)
    y = TestTensor(y_shape, None, dtype, device)
    if inplace == Inplace.INPLACE:
        d = y
    else:
        d = TestTensor(y_shape, None, dtype, device)
    ans = linearFunction(
        bias.torch_tensor() if bias_exit else None,
        x.torch_tensor(),
        w.torch_tensor()
    )
    w_data_t = w.actual_tensor().clone().t().contiguous()
    w_t = TestTensor((N, K), w_data_t.stride(), dtype, device, mode="manual", set_tensor=w_data_t)
    w_packed, w_scale, w_zero = per_token_quant_int8_torch(w_data_t, symmetric)
    # w_packed, w_scale, w_zero = computeQuant(
    #     handle,
    #     device,
    #     w_t, 
    #     symmetric,
    #     sync=None)
    
    weights = TestTensor(
        w_shape, w_packed.t().contiguous().stride(), InfiniDtype.I8, device, mode="manual", set_tensor=w_packed.t().contiguous()
    )
    weights_scale = TestTensor(
        (1, N), w_scale.t().contiguous().stride(), InfiniDtype.F32, device, mode="manual", set_tensor=w_scale.t().contiguous()
    )
    if symmetric:
        weights_zero = None
    else:
        weights_zero = TestTensor(
            (1, N), w_zero.t().contiguous().stride(), InfiniDtype.F32, device, mode="manual", set_tensor=w_zero.t().contiguous()
        )
    x_p, x_s, x_z = per_token_quant_int8_torch(x.actual_tensor(), symmetric)
    # x_p, x_s, x_z = computeQuant(
    #     handle,
    #     device,
    #     x, 
    #     symmetric,
    #     sync=None)
    
    x_packed = TestTensor(
        x_shape, x_p.stride(), InfiniDtype.I8, device, mode="manual", set_tensor=x_p
    )
    x_scale = TestTensor((M, 1), x_s.stride(), InfiniDtype.F32, device, mode="manual", set_tensor=x_s)
    if symmetric:
        x_zero = None
    else:
        x_zero = TestTensor((M, 1), x_z.stride(), InfiniDtype.F32, device, mode="manual", set_tensor=x_z)
    
    
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateI8GemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            bias.descriptor if bias_exit else None,
            x_packed.descriptor,
            x_scale.descriptor,
            weights.descriptor,
            weights_scale.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    
    y.destroy_desc()
    
    if bias_exit:
        bias.destroy_desc()
    x_packed.destroy_desc()
    x_scale.destroy_desc()
    if symmetric == False:
        x_zero.destroy_desc()
    weights.destroy_desc()
    weights_scale.destroy_desc()
    if symmetric == False:
        weights_zero.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetI8GemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x_packed.device)

    def lib_linear():
        check_error(
            LIBINFINIOP.infiniopI8Gemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                bias.data() if bias_exit else None,
                x_packed.data(),
                x_scale.data(),
                weights.data(),
                weights_scale.data(),
                None,
            )
        )

    lib_linear()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    print(max(abs(y.actual_tensor() - ans).flatten()))
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: linearFunction(bias.torch_tensor() if bias_exit else None, x.torch_tensor(), w.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_linear(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyI8GemmDescriptor(descriptor))
    

if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
