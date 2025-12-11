# ============================================================
# 0. infinicore 包导入，配置测试用 safetensors 临时存储路径
# ============================================================
import os
import sys


import torch
import torch.nn as nn

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python/infinicore"))
)

save_dir = os.path.join(os.path.dirname(__file__), "../../tmp")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "infinicore_parameter_test.safetensors")


import infinicore  # noqa: E402
from infinicore.nn import Module, Parameter  # noqa: E402

device_str = "cuda"


class InfiniCoreParameterNet(Module):
    def __init__(self):
        super().__init__()
        self.a = infinicore.nn.Parameter(
            infinicore.empty(
                (1, 2, 3), dtype=infinicore.float32, device=infinicore.device("cpu", 0)
            )
        )

    def forward(self, x):
        return infinicore.add(self.a, x)


infinicore_model_infer = InfiniCoreParameterNet()
# ============================================================
# 2. 加载权重
# ============================================================
params_dict = {
    "a": infinicore.empty(
        (1, 2, 3), dtype=infinicore.float32, device=infinicore.device(device_str, 0)
    )
}
infinicore_model_infer.load_state_dict(params_dict)


# ============================================================
# 3. 计算
# ============================================================
x = infinicore.empty(
    (1, 2, 3), dtype=infinicore.float32, device=infinicore.device(device_str, 0)
)

infinicore_model_out = infinicore_model_infer(x)
ref_out = infinicore.add(params_dict["a"], x)

# ============================================================
# 4. 对比结果
# ============================================================
print("InfiniCoreModule 与 Torch (CPU) 最大误差: 手动查看 ")
infinicore_model_out.debug()
ref_out.debug()


# ============================================================
# 5. 测试 Parameter 的基本功能
# ============================================================

print("\n=== 测试 Parameter 基本功能 ===")

# 测试 1: 创建 Parameter
param1 = infinicore.nn.Parameter(
    infinicore.empty(
        (1, 2, 3), dtype=infinicore.float32, device=infinicore.device(device_str, 0)
    )
)
print(f"✓ 创建 Parameter，形状: {param1.shape}")
# 检查是否是 Parameter 类型（可能是 InfiniCoreParameter 的别名）

assert isinstance(param1, infinicore.nn.Parameter), "应该是 Parameter 类型"
assert isinstance(param1, infinicore.Tensor), "应该是 torch.Tensor 的子类"


# 测试 3: 自动注册到 Module
class TestModule(Module):
    def __init__(self):
        super().__init__()
        self.weight = infinicore.nn.Parameter(
            infinicore.empty(
                (1, 2, 3),
                dtype=infinicore.float32,
                device=infinicore.device(device_str),
            )
        )
        self.bias = infinicore.nn.Parameter(
            infinicore.empty(
                (1, 2, 3),
                dtype=infinicore.float32,
                device=infinicore.device(device_str),
            )
        )


test_module = TestModule()
param_count = sum(1 for _ in test_module.parameters())
print(f"✓ 自动注册到 Module，参数数量: {param_count}")
assert param_count == 2, f"应该有 2 个参数，实际为 {param_count}"

# 测试 4: 参数访问
assert test_module.weight is not None, "weight 应该可以访问"
assert test_module.bias is not None, "bias 应该可以访问"
print("✓ 参数可以通过属性访问")

# 测试 5: state_dict
state_dict = test_module.state_dict()
print(f"✓ state_dict 键数量: {len(state_dict)}")
assert "weight" in state_dict, "state_dict 应该包含 weight"
assert "bias" in state_dict, "state_dict 应该包含 bias"
print(f"✓ state_dict 键: {list(state_dict.keys())}")

# 测试 6: __repr__
repr_str = repr(param1)
print(f"✓ __repr__ 方法: 输出包含类名")
assert "Parameter" in repr_str or "InfiniCoreParameter" in repr_str, "repr 应该包含类名"
print(repr_str[:100] + "...")


# 测试 9: 从 None 创建
# param_empty = Parameter(None)
# print(f"✓ 从 None 创建 Parameter，形状: {param_empty.shape}")
# assert param_empty.shape == torch.Size([0]), "从 None 创建应该是空张量"


# 测试 10: 深拷贝
# import copy

# param_copy = copy.deepcopy(param1)
# print(f"✓ 深拷贝 Parameter，形状: {param_copy.shape}")
# assert param_copy.shape == param1.shape, "深拷贝后形状应该相同"
# assert not torch.equal(param_copy, param1) or id(param_copy) != id(param1), (
#     "深拷贝应该是新对象"
# )

print("\n=== 所有测试通过! ===")
