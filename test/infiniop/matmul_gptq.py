import torch
import torch.nn as nn
import math
import ctypes
from ctypes import POINTER, Structure, c_int32, c_size_t, c_uint64, c_void_p, c_float
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    create_workspace,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = []

MODELS = {
    "7B": [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
    # "13B": [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
    # "33B": [(6656, 3 * 6656), (6656, 6656), (6656, 2 * 17664), (17664, 6656)],
    # "70B": [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
}

# Loop through models and layers to generate the new _TEST_CASES
for _, layers in MODELS.items():
    for layer in layers:
        for batch in [1, 16]:
            _TEST_CASES.append(((batch, layer[0], layer[1])))


# Data types used for testing
_TENSOR_DTYPES = [torch.float16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# ==============================================================================
#  Definitions
# ==============================================================================
class MatmulGptqDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopMatmulGptqDescriptor_t = POINTER(MatmulGptqDescriptor)


# PyTorch implementation for matrix multiplication


# The argument list should be (lib, handle, torch_device, <param list>, dtype)
# The <param list> should keep the same order as the one specified in _TEST_CASES
def test(
    lib,
    handle,
    torch_device,
    M,
    K,
    N,
    dtype=torch.float16,
):
    print(
        f"Testing MatmulGptq on {torch_device}" f" M:{M}, K:{K}, N:{N}, dtype:{dtype}"
    )
    torch.manual_seed(12)
    # Initialize tensors
    a = 1e0 * torch.randn([M, K], dtype=dtype).to(torch_device)
    layer = nn.Linear(K, N)
    b = 1e-3 * layer.weight.data.to(dtype).to(torch_device)
    c = torch.zeros([M, N], dtype=dtype).to(torch_device).t()
    packed_weights = torch.zeros([N, K // 8], dtype=torch.int32).to(torch_device)

    group_size = -1
    num_groups = 1
    if group_size == -1:
        num_groups = 1
    else:
        num_groups = K // group_size

    s = torch.zeros([N, num_groups], dtype=dtype).to(torch_device)
    z = torch.zeros([N, num_groups], dtype=dtype).to(torch_device)

    ans = torch.matmul(b.to(torch.float32), a.t().to(torch.float32)).to(dtype)

    a_tensor, b_tensor, c_tensor, s_tensor, z_tensor, packed_weights_tensor = (
        to_tensor(a.t(), lib),
        to_tensor(b, lib),
        to_tensor(c, lib),
        to_tensor(s, lib),
        to_tensor(z, lib),
        to_tensor(packed_weights, lib),
    )

    descriptor = infiniopMatmulGptqDescriptor_t()
    check_error(
        lib.infiniopCreateMatmulGptqDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            a_tensor.descriptor,
            packed_weights_tensor.descriptor,
            s_tensor.descriptor,
            z_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [
        a_tensor,
        b_tensor,
        c_tensor,
        s_tensor,
        z_tensor,
        packed_weights_tensor,
    ]:
        tensor.destroyDesc(lib)

    # Get workspace size and create workspace
    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetMatmulGptqWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, a.device)

    # Execute infiniop matmul_gptq operator
    check_error(
        lib.infiniopMatmulQuant(
            descriptor,
            workspace.data_ptr() if workspace is not None else None,
            workspace_size.value,
            packed_weights_tensor.data,
            s_tensor.data,
            z_tensor.data,
            a_tensor.data,
            b_tensor.data,
            None,
        )
    )

    def lib_matmul_gptq():
        check_error(
            lib.infiniopMatmulGptq(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                c_tensor.data,
                a_tensor.data,
                packed_weights_tensor.data,
                s_tensor.data,
                z_tensor.data,
                None,
            )
        )

    lib_matmul_gptq()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    # tmpa = ans.flatten()
    # tmpc = c.flatten()
    # for i in range(tmpa.shape[0]):
    #     if abs(tmpa[i] - tmpc[i]) > atol + rtol * abs(tmpa[i]):
    #         print(tmpa[i], tmpc[i], abs(tmpa[i] - tmpc[i]), rtol * abs(tmpa[i]))
    #         break

    if DEBUG:
        debug(c, ans, atol=atol, rtol=rtol)
    assert torch.allclose(c, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: matmul_gptq(a, b), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_matmul_gptq(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroyMatmulGptqDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateMatmulGptqDescriptor.restype = c_int32
    lib.infiniopCreateMatmulGptqDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopMatmulGptqDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetMatmulGptqWorkspaceSize.restype = c_int32
    lib.infiniopGetMatmulGptqWorkspaceSize.argtypes = [
        infiniopMatmulGptqDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopMatmulQuant.restype = c_int32
    lib.infiniopMatmulQuant.argtypes = [
        infiniopMatmulGptqDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopMatmulGptq.restype = c_int32
    lib.infiniopMatmulGptq.argtypes = [
        infiniopMatmulGptqDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyMatmulGptqDescriptor.restype = c_int32
    lib.infiniopDestroyMatmulGptqDescriptor.argtypes = [
        infiniopMatmulGptqDescriptor_t,
    ]

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
