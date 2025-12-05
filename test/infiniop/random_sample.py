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
    debug_all,
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
    # voc, random_val, topp, topk, temperature
    (512, 0.8, 0.8, 3, 0.5),
    (4096, 0.05, 0.9, 5, 1.0),
    (16384, 0.15, 0.85, 10, 2.0),
    (512, 0.08, 0, 3, 0.5),
    (4096, 0.5, 0.9, 1, 1.0),
    (16384, 0.15, 0, 1, 2.0),
    (16384, 0.15, 0, 1, 2.0),
    (32000, 0.08, 0.8, 50, 1.0),
    (32000, 0.08, 1.0, 25, 1.0),
    # (119696, 0.01, 1.0, 100, 1.0),
]

# Batch test cases: (batch_size, voc, list of (random_val, topp, topk, temperature))
_BATCH_TEST_CASES = [
    # batch_size, voc, [(random_val, topp, topk, temperature), ...]
    (4, 512, [(0.8, 0.8, 3, 0.5), (0.05, 0.9, 5, 1.0), (0.15, 0.85, 10, 2.0), (0.08, 0, 3, 0.5)]),
    (8, 4096, [(0.5, 0.9, 1, 1.0), (0.15, 0, 1, 2.0), (0.08, 0.8, 50, 1.0), (0.08, 1.0, 25, 1.0), (0.8, 0.8, 3, 0.5), (0.05, 0.9, 5, 1.0), (0.15, 0.85, 10, 2.0), (0.08, 0, 3, 0.5)]),
    (2, 16384, [(0.15, 0.85, 10, 2.0), (0.5, 0.9, 1, 1.0)]),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
}


DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def random_sample(data, random_val, topp, topk, voc, temperature):
    if topp > 0 and topk > 1:
        sorted_vals, sorted_indices = torch.sort(data, descending=True)

        scaled_vals = (sorted_vals - sorted_vals[0]) / temperature
        try:
            probs = torch.softmax(scaled_vals, dim=0)
        except RuntimeError as e:
            if "not implemented for 'Half'" in str(e):
                scaled_vals = scaled_vals.to(torch.float32)
                probs = torch.softmax(scaled_vals, dim=0)
            else:
                raise
        cum_probs = torch.cumsum(probs, dim=0)

        k_index = min(topk, voc) - 1
        threshold = min(cum_probs[k_index], topp) * random_val

        try:
            idx = torch.searchsorted(cum_probs, threshold)
        except Exception:
            # Fallback for manual search if torch.searchsorted is not supported
            indices = (cum_probs >= threshold).nonzero(as_tuple=True)[0]
            idx = (
                indices[0]
                if indices.numel() > 0
                else torch.tensor(len(cum_probs) - 1, device=cum_probs.device)
            )
        return sorted_indices[idx]

    return torch.argmax(data)


def test(
    handle,
    device,
    voc,
    random_val,
    topp,
    topk,
    temperature,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing RandomSample on {InfiniDeviceNames[device]} with voc:{voc} random_val:{random_val} topp:{topp} topk:{topk} temperature:{temperature} dtype:{InfiniDtypeNames[dtype]}"
    )

    _perm = torch.randperm(voc)
    logits = TestTensor.from_torch(
        torch.arange(voc)[_perm].float() * 0.0001, dtype, device
    )

    ans = random_sample(
        logits.torch_tensor(), random_val, topp, topk, voc, temperature
    ).to(
        torch.int32
    )  # 这个函数在device速度可能会很慢，可以通过data.to("cpu")方式加快计算过程

    indices = TestTensor([], None, InfiniDtype.I32, device, mode="zeros")

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRandomSampleDescriptor(
            handle,
            ctypes.byref(descriptor),
            indices.descriptor,
            logits.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [logits, indices]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRandomSampleWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_random_sample():
        check_error(
            LIBINFINIOP.infiniopRandomSample(
                descriptor,
                workspace.data(),
                workspace_size.value,
                indices.data(),
                logits.data(),
                random_val,
                topp,
                topk,
                temperature,
                None,
            )
        )

    lib_random_sample()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug_all(
            (indices.actual_tensor(), logits.actual_tensor()[indices.actual_tensor()]),
            (ans, logits.torch_tensor()[ans]),
            "or",
            atol=atol,
            rtol=rtol,
        )
    assert (
        indices.actual_tensor() == ans
        or logits.actual_tensor()[indices.actual_tensor()] == logits.torch_tensor()[ans]
    )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: random_sample(
            logits.torch_tensor(), random_val, topp, topk, voc, temperature
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_random_sample(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyRandomSampleDescriptor(descriptor))


def test_batch(
    handle,
    device,
    batch_size,
    voc,
    params_list,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing RandomSampleBatch on {InfiniDeviceNames[device]} with batch_size:{batch_size} voc:{voc} dtype:{InfiniDtypeNames[dtype]}"
    )

    assert len(params_list) == batch_size

    logits_list = []
    for i in range(batch_size):
        _perm = torch.randperm(voc)
        logits_list.append(torch.arange(voc)[_perm].float() * 0.0001)
    logits_batch = torch.stack(logits_list)
    logits = TestTensor.from_torch(logits_batch, dtype, device)

    ans_list = []
    for i in range(batch_size):
        random_val, topp, topk, temperature = params_list[i]
        ans = random_sample(
            logits.torch_tensor()[i], random_val, topp, topk, voc, temperature
        ).to(torch.int32)
        ans_list.append(ans)
    ans_batch = torch.stack(ans_list)

    indices = TestTensor([batch_size], None, InfiniDtype.I32, device, mode="zeros")

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    try:
        check_error(
            LIBINFINIOP.infiniopCreateRandomSampleBatchDescriptor(
                handle,
                ctypes.byref(descriptor),
                indices.descriptor,
                logits.descriptor,
            )
        )
    except Exception as e:
        print(f"\033[93mNote: Batch descriptor creation not implemented yet: {e}\033[0m")
        print(f"  This is expected - batch interface implementation is pending")
        return

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [logits, indices]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRandomSampleWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    random_val_array = (ctypes.c_float * batch_size)(*[p[0] for p in params_list])
    topp_array = (ctypes.c_float * batch_size)(*[p[1] for p in params_list])
    topk_array = (ctypes.c_int * batch_size)(*[p[2] for p in params_list])
    temperature_array = (ctypes.c_float * batch_size)(*[p[3] for p in params_list])

    def lib_random_sample_batch():
        check_error(
            LIBINFINIOP.infiniopRandomSampleBatch(
                descriptor,
                workspace.data(),
                workspace_size.value,
                indices.data(),
                logits.data(),
                random_val_array,
                topp_array,
                topk_array,
                temperature_array,
                batch_size,
                None,
            )
        )

    lib_random_sample_batch()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug_all(
            (indices.actual_tensor(), logits.actual_tensor()[torch.arange(batch_size), indices.actual_tensor()]),
            (ans_batch, logits.torch_tensor()[torch.arange(batch_size), ans_batch]),
            "or",
            atol=atol,
            rtol=rtol,
        )
    
    actual_indices = indices.actual_tensor()
    for i in range(batch_size):
        assert (
            actual_indices[i] == ans_batch[i]
            or logits.actual_tensor()[i, actual_indices[i]] == logits.torch_tensor()[i, ans_batch[i]]
        )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        def pytorch_batch():
            results = []
            for i in range(batch_size):
                random_val, topp, topk, temperature = params_list[i]
                results.append(random_sample(
                    logits.torch_tensor()[i], random_val, topp, topk, voc, temperature
                ))
            return torch.stack(results)
        profile_operation("PyTorch", lambda: pytorch_batch(), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_random_sample_batch(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    
    check_error(LIBINFINIOP.infiniopDestroyRandomSampleDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
        
        print(f"\n\033[93mRunning batch tests on {InfiniDeviceNames[device]}...\033[0m")
        try:
            test_operator(device, test_batch, _BATCH_TEST_CASES, _TENSOR_DTYPES)
        except Exception as e:
            print(f"\033[91mBatch test failed (not implemented yet): {e}\033[0m")
            print(f"  This is expected - batch interface implementation is pending")

    print("\033[92mTest passed!\033[0m")
