import torch
import ctypes
from ctypes import c_uint64
import numpy as np

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
    infiniopOperatorDescriptor_t,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    TestWorkspace,
    InfiniDeviceEnum,
)
from torch.nn import functional as F

_TEST_CASES = [
    # Single sample classification
    ((10,), 10),
    ((200,), 200),
    # 2D: (N, C) - batch classification
    ((4, 10), 10),
    ((8, 5), 5),
    ((16, 100), 100),
    ((32, 1000), 1000),
    ((64, 21), 21),
    ((128, 50), 50),
    # 3D: (N, C, d1) - sequence classification
    ((4, 10, 5), 10),
    # 4D: (N, C, d1, d2) - image segmentation
    ((2, 8, 8, 8), 8),
    # 5D: (N, C, d1, d2, d3) - 3D segmentation
    ((3, 10, 10, 20, 30), 10),
]

_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16]
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}
DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def cross_entropy_loss_pytorch(logits, target):
    return F.cross_entropy(logits.double(), target.long(), reduction="mean")


def test(
    handle,
    device,
    input_shape,
    num_classes,
    tensor_dtype=InfiniDtype.F32,
    sync=None,
):
    # 根据输入形状确定logits和target的形状
    if len(input_shape) == 1:
        # Shape (C,) - single sample classification
        logits_shape = (num_classes,)
        target_shape = (1,)  # 修改：使用 (1,) 而不是标量
    else:
        # Shape (N, C, [d1], [d2], ...)
        logits_shape = input_shape
        target_shape = (input_shape[0],) + input_shape[2:]

    print(
        f"Testing CrossEntropyLoss on {InfiniDeviceNames[device]} with logits_shape: {logits_shape}, target_shape: {target_shape}, dtype:{InfiniDtypeNames[tensor_dtype]}"
    )

    # 创建logits张量
    logits = TestTensor(logits_shape, None, dt=tensor_dtype, device=device)

    # 创建target张量
    target_torch = torch.randint(
        0,
        num_classes,
        target_shape,
        dtype=torch.long,
        device=logits.torch_tensor().device,
    )
    target = TestTensor.from_torch(target_torch, dt=InfiniDtype.I64, device=device)

    # 创建loss张量
    loss = TestTensor((1,), None, dt=tensor_dtype, device=device)

    # 计算PyTorch参考损失
    if len(input_shape) == 1:
        # 对于一维logits，target需要是标量
        target_scalar = target.torch_tensor()[0]
        pytorch_loss = cross_entropy_loss_pytorch(logits.torch_tensor(), target_scalar)
    else:
        pytorch_loss = cross_entropy_loss_pytorch(
            logits.torch_tensor(), target.torch_tensor()
        )

    # 将参考结果存储到loss张量
    loss.torch_tensor()[0] = pytorch_loss.to(loss.torch_tensor().dtype)

    if sync:
        sync()

    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateCrossEntropyLossDescriptor(
            handle,
            ctypes.byref(descriptor),
            loss.descriptor,
            logits.descriptor,
            target.descriptor,
        )
    )

    # 销毁tensor的描述符以防止内核直接使用
    for tensor in [logits, target, loss]:
        tensor.destroy_desc()

    # 获取工作空间大小并创建工作空间
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetCrossEntropyLossWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # PyTorch参考实现函数
    def torch_cross_entropy():
        if len(input_shape) == 1:
            target_scalar = target.torch_tensor()[0]
            result = cross_entropy_loss_pytorch(logits.torch_tensor(), target_scalar)
        else:
            result = cross_entropy_loss_pytorch(
                logits.torch_tensor(), target.torch_tensor()
            )
        loss.torch_tensor()[0] = result.to(loss.torch_tensor().dtype)

    # InfiniOP实现函数
    def lib_cross_entropy():
        check_error(
            LIBINFINIOP.infiniopCrossEntropyLoss(
                descriptor,
                workspace.data(),
                workspace_size.value,
                loss.data(),
                logits.data(),
                target.data(),
                None,
            )
        )

    # 执行InfiniOP算子
    lib_cross_entropy()

    if sync:
        sync()

    # 验证结果
    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    actual_loss = loss.actual_tensor()[0]
    expected_loss = loss.torch_tensor()[0]

    if DEBUG:
        print(f"Expected loss: {expected_loss.item()}")
        print(f"Actual loss: {actual_loss.item()}")
        if target_shape:
            print(
                f"Target shape: {target_shape}, first few targets: {target.torch_tensor().flatten()[:5]}"
            )
        else:
            print(f"Target (scalar): {target.torch_tensor()[0].item()}")
        debug(actual_loss, expected_loss, atol=atol, rtol=rtol)

    if not torch.allclose(actual_loss, expected_loss, atol=atol, rtol=rtol):
        print("--- ERROR ANALYSIS ---")
        print(f"Expected: {expected_loss.item()}, Actual: {actual_loss.item()}")
        print(f"Difference: {abs(actual_loss - expected_loss).item()}")
        print(f"Tolerance: atol={atol}, rtol={rtol}")

    assert torch.allclose(actual_loss, expected_loss, atol=atol, rtol=rtol)

    # Profile功能
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_cross_entropy(), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_cross_entropy(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyCrossEntropyLossDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mAll CrossEntropyLoss tests passed!\033[0m")
