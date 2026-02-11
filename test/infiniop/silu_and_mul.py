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
# Format: (input_shape, output_shape)
# Referencing vLLM kernel Silu_and_Mul interface:
# input_shape is [..., 2*d], output_shape is [..., d]
_TEST_CASES = [
    # input_shape, output_shape
    ((2, 8), (2, 4)),
    ((1024, 1024), (1024, 512)),
    ((16, 8192), (16, 4096)),
    ((2, 128, 2048), (2, 128, 1024)),
    ((8, 1, 4096), (8, 1, 2048)),
    ((2, 4, 16, 256), (2, 4, 16, 128)),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


# PyTorch reference: silu(gate) * up where [gate, up] = split(input)
def silu_and_mul_torch(out, input_tensor):
    """
    Computes the SwiGLU activation function: SiLU(gate) * up.
    """
    # Split the last dimension into two halves:
    # the first half is 'gate', the second is 'up'
    d = input_tensor.shape[-1] // 2
    gate = input_tensor[..., :d]
    up = input_tensor[..., d:]

    # Apply SiLU to the gate and multiply by the up projection
    torch.mul(torch.nn.functional.silu(gate), up, out=out)


# ==============================================================================
#  Test Logic
# ==============================================================================
def test(
    handle,
    device,
    input_shape,
    output_shape,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing SiluAndMul on {InfiniDeviceNames[device]} with "
        f"input_shape:{input_shape} output_shape:{output_shape} dtype:{InfiniDtypeNames[dtype]}"
    )

    a = TestTensor(input_shape, None, dtype, device)
    c = TestTensor(output_shape, None, dtype, device, mode="zeros")
    ans = TestTensor(output_shape, None, dtype, device, mode="zeros")

    # Only support contiguous Tensor
    if not (
        a.torch_tensor().is_contiguous()
        and c.torch_tensor().is_contiguous()
        and ans.torch_tensor().is_contiguous()
    ):
        raise ValueError("This operator only supports contiguous memory layout.")

    # PyTorch answer reference
    def torch_silu_and_mul_reference():
        silu_and_mul_torch(ans.torch_tensor(), a.torch_tensor())

    torch_silu_and_mul_reference()

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSiluAndMulDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            a.descriptor,
        )
    )

    for tensor in [a, c]:
        tensor.destroy_desc()

    # Workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSiluAndMulWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_op():
        check_error(
            LIBINFINIOP.infiniopSiluAndMul(
                descriptor,
                workspace.data(),
                workspace_size.value,
                c.data(),
                a.data(),
                None,
            )
        )

    lib_op()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    if DEBUG:
        debug(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: torch_silu_and_mul_reference(),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_op(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroySiluAndMulDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mSiluAndMul Test passed!\033[0m")
