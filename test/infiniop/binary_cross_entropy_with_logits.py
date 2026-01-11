import torch
import ctypes
from ctypes import c_uint64, c_float, c_char_p
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

# ==============================================================================
# Configuration
# ==============================================================================
# 测试场景：(shape, has_weight, has_pos_weight, reduction)
_TEST_CASES_DATA = [
    ((4, 5), False, False, "none"),
    ((8, 8), True, False, "sum"),
    ((32, 512), False, True, "mean"),
    ((16, 32, 64), True, True, "mean"), 
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 5e-2},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

_REDUCTIONS = ["none", "mean", "sum"]

_REDUCTION_MAP = {
    "none": 0,  # INFINIOP_REDUCTION_NONE
    "mean": 1,  # INFINIOP_REDUCTION_MEAN
    "sum": 2,   # INFINIOP_REDUCTION_SUM
}

# 生成最终测试用例组合
_TEST_CASES = _TEST_CASES_DATA 

DEBUG = False
PROFILE = False

def test(
    handle,
    device,
    shape,
    has_weight=False,
    has_pos_weight=False,
    reduction="none",
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing BCEWithLogits on {InfiniDeviceNames[device]} shape:{shape} "
        f"weight:{has_weight} pos_weight:{has_pos_weight} reduction:{reduction} dtype:{InfiniDtypeNames[dtype]}"
    )

    # 1. 准备输入 Tensor
    input_tensor = TestTensor(shape, None, dtype, device)
    target = TestTensor(shape, None, dtype, device)
    
    weight = TestTensor(shape, None, dtype, device) if has_weight else None
    # pos_weight 通常在最后一维广播，形状为 (C,)
    pos_weight_shape = (shape[-1],)
    pos_weight = TestTensor(pos_weight_shape, None, dtype, device) if has_pos_weight else None

    # 2. 使用 PyTorch 计算参考答案
    torch_input = input_tensor.torch_tensor()
    torch_target = target.torch_tensor()
    torch_weight = weight.torch_tensor() if has_weight else None
    torch_pos_weight = pos_weight.torch_tensor() if has_pos_weight else None

    ans = torch.nn.functional.binary_cross_entropy_with_logits(
        torch_input, 
        torch_target, 
        weight=torch_weight, 
        pos_weight=torch_pos_weight, 
        reduction=reduction
    )

    # 3. 准备输出 Tensor (根据 reduction 确定形状)
    out_shape = () if reduction != "none" else shape
    out = TestTensor(out_shape, None, dtype, device)

    if sync is not None:
        sync()

    # 4. 创建描述符并执行
    descriptor = infiniopOperatorDescriptor_t()
    
    # 模拟 C 接口调用
    reduction_enum = _REDUCTION_MAP[reduction]
    check_error(
        LIBINFINIOP.infiniopCreateBCEWithLogitsDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            input_tensor.descriptor,
            target.descriptor,
            weight.descriptor if has_weight else None,
            pos_weight.descriptor if has_pos_weight else None,
            reduction_enum  # 传入归约方式枚举值，对应 infiniopReduction_t
        )
    )

    # 销毁临时描述符
    for t in [input_tensor, target, out]:
        t.destroy_desc()
    if weight: weight.destroy_desc()
    if pos_weight: pos_weight.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetBCEWithLogitsWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_op():
        check_error(
            LIBINFINIOP.infiniopBCEWithLogits(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                input_tensor.data(),
                target.data(),
                weight.data() if has_weight else None,
                pos_weight.data() if has_pos_weight else None,
                None,
            )
        )

    lib_op()

    if sync is not None:
        sync()

    # 5. 验证结果
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation("PyTorch", lambda: torch.nn.functional.binary_cross_entropy_with_logits(
            torch_input, torch_target, weight=torch_weight, pos_weight=torch_pos_weight, reduction=reduction
        ), device)
        profile_operation("   lib", lib_op, device)

    check_error(LIBINFINIOP.infiniopDestroyBCEWithLogitsDescriptor(descriptor))

if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mBCEWithLogits tests passed!\033[0m")
