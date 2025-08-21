import ctypes
import sys

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "test/infiniop")

from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceEnum,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    create_handle,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
    to_torch_dtype,
)

import infini


def add(a_, b_):
    c_ = torch.empty_like(a_)

    LIBINFINIOP.infinirtSetDevice(InfiniDeviceEnum.NVIDIA, ctypes.c_int(0))

    handle = create_handle()

    descriptor = infiniopOperatorDescriptor_t()

    # No hard-coding.
    a = TestTensor.from_torch(a_, InfiniDtype.F16, InfiniDeviceEnum.NVIDIA)
    b = TestTensor.from_torch(b_, InfiniDtype.F16, InfiniDeviceEnum.NVIDIA)
    c = TestTensor.from_torch(c_, InfiniDtype.F16, InfiniDeviceEnum.NVIDIA)

    check_error(
        LIBINFINIOP.infiniopCreateAddDescriptor(
            handle, ctypes.byref(descriptor), c.descriptor, a.descriptor, b.descriptor
        )
    )

    workspace_size = ctypes.c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAddWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    check_error(
        LIBINFINIOP.infiniopAdd(
            descriptor,
            workspace.data(),
            workspace.size(),
            c.data(),
            a.data(),
            b.data(),
            None,
        )
    )

    return c.actual_tensor()


if __name__ == "__main__":
    torch.manual_seed(0)

    infini.device("cuda:0")
    x = infini.empty(10, 10, dtype=InfiniDtype.F16, device="cuda:0")
    print(x.shape)
    print(x.strides)

    size = 98432
    dtype = torch.float16
    device = "cuda"

    input = torch.randn(size, dtype=dtype, device=device)
    other = torch.randn(size, dtype=dtype, device=device)

    infini_output = add(input, other)
    torch_output = torch.add(input, other)

    print(infini_output)
    print(torch_output)

    if torch.allclose(infini_output, torch_output):
        print("✅ Infini and PyTorch match.")
    else:
        print("❌ Infini and PyTorch differ.")
