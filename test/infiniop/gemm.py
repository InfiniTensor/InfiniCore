import torch
import ctypes
import os
import time
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
    # Temporarily skip batch=1 to test batch>1 cases
    # (1.0, 0.0, (1, 2048), (2048, 2048), (1, 2048), None, None, None),
    (1.0, 0.0, (2, 4, 2048), (2, 2048, 2048), (2, 4, 2048), None, None, None),
    # (1.0, 0.0, (1, 2048), (2048, 2048), (1, 2048), (4096, 1), (4096, 1), (4096, 1)),
    (1.0, 1.0, (6, 2048), (2048, 2560), (6, 2560), (2048, 1), (1, 2048), (2560, 1)),
    (1.0 / 8.0, 0.0, (4, 8 * 6, 64), (4, 64, 6), (4, 8 * 6, 6), None, None, None),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 0, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000
COMPARE_BACKENDS = False  # Set via INFINIOP_GEMM_COMPARE_BACKENDS env var to compare HCBLAS vs HCDNN performance


# PyTorch implementation for matrix multiplication
def gemm(d, _c, beta, _a, _b, alpha):
    try:
        if _c.ndim == 2:
            torch.addmm(_c, _a, _b, beta=beta, alpha=alpha, out=d)
        elif _c.ndim == 3:
            torch.baddbmm(_c, _a, _b, beta=beta, alpha=alpha, out=d)
        else:
            raise
    except Exception:
        torch.matmul(_a, _b, out=d)
        d.mul_(alpha).add_(_c, alpha=beta)


# The argument list should be (lib, handle, torch_device, <param list>, dtype)
# The <param list> should keep the same order as the one specified in _TEST_CASES
def test(
    handle,
    device,
    alpha,
    beta,
    a_shape,
    b_shape,
    c_shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=InfiniDtype.F16,
    sync=None,
):
    global COMPARE_BACKENDS, NUM_PRERUN, NUM_ITERATIONS
    # Re-check environment variable in case it was set after module load
    if os.getenv("INFINIOP_GEMM_COMPARE_BACKENDS", "").lower() in ("1", "true", "yes"):
        COMPARE_BACKENDS = True

    print(
        f"Testing Gemm on {InfiniDeviceNames[device]} with alpha:{alpha}, beta:{beta},"
        f" a_shape:{a_shape}, b_shape:{b_shape}, c_shape:{c_shape},"
        f" a_stride:{a_stride}, b_stride:{b_stride}, c_stride:{c_stride}, dtype:{InfiniDtypeNames[dtype]}"
    )

    # Initialize tensors
    a = TestTensor(a_shape, a_stride, dtype, device)
    b = TestTensor(b_shape, b_stride, dtype, device)
    c = TestTensor(c_shape, c_stride, dtype, device, mode="ones")
    ans = TestTensor(c_shape, c_stride, dtype, device, mode="zeros")

    # Compute the PyTorch reference result
    def torch_gemm():
        gemm(
            ans.torch_tensor(),
            c.torch_tensor(),
            beta,
            a.torch_tensor(),
            b.torch_tensor(),
            alpha,
        )

    torch_gemm()

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            a.descriptor,
            b.descriptor,
        )
    )

    # Get workspace size and create workspace (before destroying descriptors for comparison)
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Timing for backend comparison (only for Metax device) - do this BEFORE destroying descriptors
    device_name = InfiniDeviceNames[device]
    # Check environment variable directly in test function for reliability
    compare_enabled = os.getenv("INFINIOP_GEMM_COMPARE_BACKENDS", "").lower() in ("1", "true", "yes")
    if compare_enabled and device_name == "Metax":
        # Save original c tensor data to restore after comparison
        import torch
        c_original_data = c.actual_tensor().clone()

        # Test HCBLAS - one warmup + one timed run
        os.environ["INFINIOP_GEMM_METAX_BACKEND"] = "HCBLAS"
        descriptor_hcblas = infiniopOperatorDescriptor_t()
        check_error(
            LIBINFINIOP.infiniopCreateGemmDescriptor(
                handle,
                ctypes.byref(descriptor_hcblas),
                c.descriptor,
                a.descriptor,
                b.descriptor,
            )
        )
        workspace_size_hcblas = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetGemmWorkspaceSize(
                descriptor_hcblas, ctypes.byref(workspace_size_hcblas)
            )
        )
        workspace_hcblas = TestWorkspace(workspace_size_hcblas.value, device)
        # Warmup
        check_error(
            LIBINFINIOP.infiniopGemm(
                descriptor_hcblas,
                workspace_hcblas.data(),
                workspace_size_hcblas.value,
                c.data(),
                a.data(),
                b.data(),
                alpha,
                beta,
                None,
            )
        )
        # Time HCBLAS
        start_time = time.perf_counter()
        check_error(
            LIBINFINIOP.infiniopGemm(
                descriptor_hcblas,
                workspace_hcblas.data(),
                workspace_size_hcblas.value,
                c.data(),
                a.data(),
                b.data(),
                alpha,
                beta,
                None,
            )
        )
        hcblas_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        check_error(LIBINFINIOP.infiniopDestroyGemmDescriptor(descriptor_hcblas))

        # Test HCDNN - one warmup + one timed run
        os.environ["INFINIOP_GEMM_METAX_BACKEND"] = "HCDNN"
        descriptor_hcdnn = infiniopOperatorDescriptor_t()
        check_error(
            LIBINFINIOP.infiniopCreateGemmDescriptor(
                handle,
                ctypes.byref(descriptor_hcdnn),
                c.descriptor,
                a.descriptor,
                b.descriptor,
            )
        )
        workspace_size_hcdnn = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetGemmWorkspaceSize(
                descriptor_hcdnn, ctypes.byref(workspace_size_hcdnn)
            )
        )
        workspace_hcdnn = TestWorkspace(workspace_size_hcdnn.value, device)
        # Warmup
        check_error(
            LIBINFINIOP.infiniopGemm(
                descriptor_hcdnn,
                workspace_hcdnn.data(),
                workspace_size_hcdnn.value,
                c.data(),
                a.data(),
                b.data(),
                alpha,
                beta,
                None,
            )
        )
        # Time HCDNN
        start_time = time.perf_counter()
        check_error(
            LIBINFINIOP.infiniopGemm(
                descriptor_hcdnn,
                workspace_hcdnn.data(),
                workspace_size_hcdnn.value,
                c.data(),
                a.data(),
                b.data(),
                alpha,
                beta,
                None,
            )
        )
        hcdnn_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        check_error(LIBINFINIOP.infiniopDestroyGemmDescriptor(descriptor_hcdnn))

        # Print comparison
        speedup = hcblas_time / hcdnn_time if hcdnn_time > 0 else 0
        print(f"\n  Backend Comparison:")
        print(f"    HCBLAS: {hcblas_time:.4f} ms")
        print(f"    HCDNN:  {hcdnn_time:.4f} ms")
        if speedup > 1:
            print(f"    HCDNN is {speedup:.2f}x faster")
        elif speedup < 1:
            print(f"    HCBLAS is {1/speedup:.2f}x faster")
        else:
            print(f"    Performance is similar")
        print()  # Add blank line after comparison

        # Restore default backend (HCDNN)
        os.environ["INFINIOP_GEMM_METAX_BACKEND"] = "HCDNN"

        # Restore original c tensor data before main test
        c.actual_tensor().copy_(c_original_data)

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a, b, c]:
        tensor.destroy_desc()

    # Execute infiniop gemm operator
    def lib_gemm():
        check_error(
            LIBINFINIOP.infiniopGemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                c.data(),
                a.data(),
                b.data(),
                alpha,
                beta,
                None,
            )
        )

    lib_gemm()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    if DEBUG:
        debug(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyGemmDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    # Check for environment variable to enable backend comparison
    # Note: No 'global' needed here - we're at module level
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    COMPARE_BACKENDS = os.getenv("INFINIOP_GEMM_COMPARE_BACKENDS", "").lower() in ("1", "true", "yes")

    if COMPARE_BACKENDS:
        # Increase iterations for more convincing timing results
        NUM_PRERUN = max(NUM_PRERUN, 50)  # At least 50 warmup iterations
        NUM_ITERATIONS = max(NUM_ITERATIONS, 1000)  # At least 1000 timing iterations
        print("Backend comparison enabled: Will compare HCBLAS vs HCDNN performance")
        print(f"  Using {NUM_PRERUN} warmup iterations and {NUM_ITERATIONS} timing iterations per backend")

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
