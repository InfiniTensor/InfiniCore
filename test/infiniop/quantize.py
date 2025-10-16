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
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # alpha, beta, a_shape, b_shape, c_shape, a_stride, b_stride, c_stride
    ((1, 2048), (2048, 2048), (1, 2048), None, None, None),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 2e-1},
    # InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 0, "rtol": 2e-1},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# The argument list should be (lib, handle, torch_device, <param list>, dtype)
# The <param list> should keep the same order as the one specified in _TEST_CASES
def test(
    handle,
    device,
    a_shape,
    b_shape,
    c_shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        # f"Testing Gemm on {InfiniDeviceNames[device]} with alpha:{alpha}, beta:{beta},"
        f" a_shape:{a_shape}, b_shape:{b_shape}, c_shape:{c_shape},"
        f" a_stride:{a_stride}, b_stride:{b_stride}, c_stride:{c_stride}, dtype:{InfiniDtypeNames[dtype]}"
    )

    input = TestTensor((512, 512), None, InfiniDtype.F16, device, mode="random")
    output_q = TestTensor((512, 512), None, InfiniDtype.F8E4M3, device, mode="zeros")
    outpus_s = TestTensor((5, 512), None, InfiniDtype.F32, device, mode="zeros")

    def eval_1x128(x_quant, x_scale):
        scale = torch.repeat_interleave(x_scale, 128, dim=0)
        scale = scale[: x_quant.shape[0], : x_quant.shape[1]]

        assert (
            scale.shape == x_quant.shape
        ), f"scale shape {scale.shape} not match x_quant shape {x_quant.shape}"

        x_qdq = x_quant.to(torch.float32) * scale
        return x_qdq.to(torch.float32)

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateQuantizeDescriptor(
            handle,
            ctypes.byref(descriptor),
            input.descriptor,
            output_q.descriptor,
            outpus_s.descriptor,
        )
    )

    # Get workspace size and create workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetQuantizeWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Execute infiniop gemm operator
    def lib_quantize():
        check_error(
            LIBINFINIOP.infiniopQuantize(
                descriptor,
                workspace.data(),
                workspace_size.value,
                input.data(),
                output_q.data(),
                outpus_s.data(),
                # zeros.data(),
                128,
                0,
                -448,
                448,
                False,
                None,
            )
        )

    lib_quantize()

    ans = eval_1x128(output_q.actual_tensor(), outpus_s.actual_tensor())
    # # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    if DEBUG:
        debug(input.torch_tensor().to(torch.float32), ans, atol=atol, rtol=rtol)
    print(ans, input.torch_tensor().to(torch.float32))

    # assert torch.allclose(ans, input.torch_tensor().to(torch.float32), atol=atol)

    # # Profiling workflow
    # if PROFILE:
    #     # fmt: off
    #     profile_operation("PyTorch", lambda: torch_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
    #     profile_operation("    lib", lambda: lib_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
    #     # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyQuantizeDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
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
