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
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ------------------------------------------------------------
# 用例配置
# ------------------------------------------------------------
_TEST_CASES_ = [
    ((2, 4, 10), None, None),        # logits shape, x_stride, y_stride
    ((1, 128, 32000), None, None),
    ((4, 512, 1000), None, None),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 2e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

# ------------------------------------------------------------
# PyTorch 参考实现
# ------------------------------------------------------------
def cross_entropy_ref(logits, target):
    vocab = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab).float()
    target_flat = target.reshape(-1).long()
    loss = torch.nn.functional.cross_entropy(logits_flat, target_flat, reduction="none")
    return loss.view(target.shape).to(logits.dtype)


def test(handle, device, shape, x_stride=None, y_stride=None, dtype=InfiniDtype.F16, sync=None):
    logits_shape = shape
    label_shape = shape[:-1]
    vocab = shape[-1]

    print(f"Testing CrossEntropy on {InfiniDeviceNames[device]} logits:{logits_shape} dtype:{InfiniDtypeNames[dtype]}")

    x = TestTensor(logits_shape, x_stride, dtype, device)
    target = TestTensor(label_shape, None, InfiniDtype.I64, device)

    # 生成有效标签
    tgt = target.torch_tensor()
    tgt.copy_(torch.randint(0, vocab, label_shape, dtype=torch.int64, device=tgt.device))
    target.actual_tensor().copy_(tgt)

    reference = cross_entropy_ref(x.torch_tensor(), target.torch_tensor())
    y = TestTensor(label_shape, y_stride, dtype, device)

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateCrossEntropyDescriptor(
            handle, ctypes.byref(descriptor), y.descriptor, x.descriptor, target.descriptor
        )
    )

    for tensor in [x, y, target]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetCrossEntropyWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
    workspace = TestWorkspace(workspace_size.value, x.device)

    def run():
        check_error(
            LIBINFINIOP.infiniopCrossEntropy(
                descriptor,
                workspace.data(),
                workspace.size(),
                y.data(),
                x.data(),
                target.data(),
                None,
            )
        )

    run()
    if sync:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    assert torch.allclose(y.actual_tensor(), reference, atol=atol, rtol=rtol)

    check_error(LIBINFINIOP.infiniopDestroyCrossEntropyDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")
