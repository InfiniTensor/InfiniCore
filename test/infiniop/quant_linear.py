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
    # x_shape, w_shape, symmetric, bias_exit, y_shape, alpha, beta
    ((8, 8), (8, 8), False, True, (8, 8), 1.0, 0.0),
    ((128, 512), (512, 1024), True, False, (128, 1024), 1.0, 0.0),
    ((128, 128), (128, 128), False, True, (128, 128), 2.0, 1.0),
    ((256, 1024), (1024, 2048), True, False, (256, 2048), 1.0, 1.0),
    ((256, 2048), (2048, 1024), False, True, (256, 1024), 1.5, 2.5),
    ((1024, 2048), (2048, 4096), True, False, (1024, 4096), 1.0, 0.0),
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


def linearFunction(c, bias, x, w, alpha, beta):
    if bias is not None:
        ans = (
            alpha * torch.matmul(x.to(torch.float32), w.to(torch.float32)).to(x.dtype)
            + beta * c
            + bias
        )
    else:
        ans = (
            alpha * torch.matmul(x.to(torch.float32), w.to(torch.float32)).to(x.dtype)
            + beta * c
        )
    return ans

def computeQuant(
        handle,
        device,
        x, 
        symmetric,
        sync=None,
):
    x_shape = x.shape
    dtype = x.dt
    M, K = x_shape

    x_packed = TestTensor(x_shape, None, InfiniDtype.I8, device, mode="zeros")
    x_scale = TestTensor((M, 1), None, dtype, device)
    if symmetric:
        x_zero = None
    else:
        x_zero = TestTensor((M, 1), None, dtype, device)
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateQuantDescriptor(
            handle,
            ctypes.byref(descriptor),
            x_packed.descriptor,
            x_scale.descriptor,
            None if symmetric else x_zero.descriptor,
            x.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel

    x_packed.destroy_desc()
    x_scale.destroy_desc()
    if symmetric == False:
        x_zero.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetQuantWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)
    
    def lib_quant():
        check_error(
            LIBINFINIOP.infiniopQuant(
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

    lib_quant()
    
    if sync is not None:
        sync()
    check_error(LIBINFINIOP.infiniopDestroyQuantDescriptor(descriptor))
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
    alpha,
    beta,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing Quant Linear on {InfiniDeviceNames[device]} with x_shape:{x_shape}, w_shape:{w_shape}, symmetric:{symmetric}, bias:{bias_exit} ,alpha:{alpha}, beta:{beta}, inplace:{inplace} dtype:{InfiniDtypeNames[dtype]}"
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
        y.torch_tensor(),
        bias.torch_tensor() if bias_exit else None,
        x.torch_tensor(),
        w.torch_tensor(),
        alpha,
        beta,
    )
    w_data_t = w.actual_tensor().clone().t().contiguous()
    w_t = TestTensor((N, K), w_data_t.stride(), dtype, device, mode="manual", set_tensor=w_data_t)
    
    w_packed, w_scale, w_zero = computeQuant(
        handle,
        device,
        w_t, 
        symmetric,
        sync=None)
    
    weights = TestTensor(
        w_shape, w_packed.t().contiguous().stride(), InfiniDtype.I8, device, mode="manual", set_tensor=w_packed.t().contiguous()
    )
    weights_scale = TestTensor(
        (1, N), w_scale.t().contiguous().stride(), dtype, device, mode="manual", set_tensor=w_scale.t().contiguous()
    )
    if symmetric:
        weights_zero = None
    else:
        weights_zero = TestTensor(
            (1, N), w_zero.t().contiguous().stride(), dtype, device, mode="manual", set_tensor=w_zero.t().contiguous()
        )
    
    x_p, x_s, x_z = computeQuant(
        handle,
        device,
        x, 
        symmetric,
        sync=None)
    
    x_packed = TestTensor(
        x_shape, x_p.stride(), InfiniDtype.I8, device, mode="manual", set_tensor=x_p
    )
    x_scale = TestTensor((M, 1), x_s.stride(), dtype, device, mode="manual", set_tensor=x_s)
    if symmetric:
        x_zero = None
    else:
        x_zero = TestTensor((M, 1), x_z.stride(), dtype, device, mode="manual", set_tensor=x_z)
    
    
    
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateLinearDescriptor(
            handle,
            ctypes.byref(descriptor),
            d.descriptor,
            y.descriptor,
            bias.descriptor if bias_exit else None,
            x_packed.descriptor,
            x_scale.descriptor,
            None if symmetric else x_zero.descriptor,
            weights.descriptor,
            weights_scale.descriptor,
            None if symmetric else weights_zero.descriptor,
            alpha,
            beta,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()
    y.destroy_desc()
    d.destroy_desc()
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
        LIBINFINIOP.infiniopGetLinearWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_linear():
        check_error(
            LIBINFINIOP.infiniopLinear(
                descriptor,
                workspace.data(),
                workspace_size.value,
                d.data(),
                y.data(),
                bias.data() if bias_exit else None,
                x_packed.data(),
                x_scale.data(),
                None if symmetric else x_zero.data(),
                weights.data(),
                weights_scale.data(),
                None if symmetric else weights_zero.data(),
                None,
            )
        )

    lib_linear()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(d.actual_tensor(), ans, atol=atol, rtol=rtol)
    
    assert torch.allclose(d.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: linearFunction(y.torch_tensor(), bias.torch_tensor() if bias_exit else None, x.torch_tensor(), w.torch_tensor(), alpha, beta), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_linear(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyLinearDescriptor(descriptor))

def compare(
    handle,
    device,
    x_shape,
    w_shape,
    symmetric,
    bias_exit,
    y_shape,
    alpha,
    beta,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F16,
    sync=None,
):
    if bias_exit or inplace == Inplace.INPLACE:
        return
    print(
        f"Compare Quant Linear with gemm on {InfiniDeviceNames[device]} with x_shape:{x_shape}, w_shape:{w_shape}, symmetric:{symmetric}, alpha:{alpha}, beta:{beta} dtype:{InfiniDtypeNames[dtype]}"
    )
    M, K = x_shape
    N = w_shape[1]
    if bias_exit:
        bias = TestTensor((N,), None, dtype, device)
    else:
        bias = None
    x = TestTensor(x_shape, None, dtype, device)
    w = TestTensor(w_shape, None, dtype, device)
    c = TestTensor(y_shape, None, dtype, device)
    c_data = c.torch_tensor().clone()
    d = TestTensor(y_shape, None, dtype, device)
    y = TestTensor(y_shape, c_data.stride(), dtype, device, mode="manual", set_tensor=c_data)
    
    w_data_t = w.actual_tensor().clone().t().contiguous()
    w_t = TestTensor((N, K), w_data_t.stride(), dtype, device, mode="manual", set_tensor=w_data_t)
    
    w_packed, w_scale, w_zero = computeQuant(
        handle,
        device,
        w_t, 
        symmetric,
        sync=None)
    
    weights = TestTensor(
        w_shape, w_packed.t().contiguous().stride(), InfiniDtype.I8, device, mode="manual", set_tensor=w_packed.t().contiguous()
    )
    weights_scale = TestTensor(
        (1, N), w_scale.t().contiguous().stride(), dtype, device, mode="manual", set_tensor=w_scale.t().contiguous()
    )
    if symmetric:
        weights_zero = None
    else:
        weights_zero = TestTensor(
            (1, N), w_zero.t().contiguous().stride(), dtype, device, mode="manual", set_tensor=w_zero.t().contiguous()
        )
    
    x_p, x_s, x_z = computeQuant(
        handle,
        device,
        x, 
        symmetric,
        sync=None)
    
    x_packed = TestTensor(
        x_shape, x_p.stride(), InfiniDtype.I8, device, mode="manual", set_tensor=x_p
    )
    x_scale = TestTensor((M, 1), x_s.stride(), dtype, device, mode="manual", set_tensor=x_s)
    if symmetric:
        x_zero = None
    else:
        x_zero = TestTensor((M, 1), x_z.stride(), dtype, device, mode="manual", set_tensor=x_z)
    
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    
    check_error(
        LIBINFINIOP.infiniopCreateLinearDescriptor(
            handle,
            ctypes.byref(descriptor),
            d.descriptor,
            c.descriptor,
            bias.descriptor if bias_exit else None,
            x_packed.descriptor,
            x_scale.descriptor,
            None if symmetric else x_zero.descriptor,
            weights.descriptor,
            weights_scale.descriptor,
            None if symmetric else weights_zero.descriptor,
            alpha,
            beta,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    
    d.destroy_desc()
    c.destroy_desc()
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
        LIBINFINIOP.infiniopGetLinearWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_linear():
        check_error(
            LIBINFINIOP.infiniopLinear(
                descriptor,
                workspace.data(),
                workspace_size.value,
                d.data(),
                c.data(),
                bias.data() if bias_exit else None,
                x_packed.data(),
                x_scale.data(),
                None if symmetric else x_zero.data(),
                weights.data(),
                weights_scale.data(),
                None if symmetric else weights_zero.data(),
                None,
            )
        )

    lib_linear()
    if PROFILE:
        # fmt: off
        profile_operation("quant_linear", lambda: lib_linear(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyLinearDescriptor(descriptor))
    #下面开始计算GEMM
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            w.descriptor,
        )
    )
    y.destroy_desc()
    x.destroy_desc()
    w.destroy_desc()
    #计算GEMM的workspace和数值结果
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Execute infiniop gemm operator
    def lib_gemm():
        check_error(
            LIBINFINIOP.infiniopGemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                w.data(),
                alpha,
                beta,
                None,
            )
        )

    lib_gemm()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), d.actual_tensor(), atol=atol, rtol=rtol)
    
    assert torch.allclose(y.actual_tensor(), d.actual_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("gemm        ", lambda: lib_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyGemmDescriptor(descriptor)) 
    

if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    for device in get_test_devices(args):
        test_operator(device, compare, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
