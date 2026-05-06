# ============================================================
# 0. infinicore 包导入，配置测试用 safetensors 临时存储路径
# ============================================================
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python/infinicore"))
)

save_dir = os.path.join(os.path.dirname(__file__), "../../tmp")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "torch_convnet_with_param.safetensors")


import infinicore  # noqa: E402
from infinicore.nn import Module  # noqa: E402


# ============================================================
# 1. 定义模型
# ============================================================
device_str = "cuda"


class InfiniCoreNet(Module):
    def __init__(self):
        super().__init__()
        self.a = infinicore.nn.Parameter(
            infinicore.empty(
                (1, 2, 3),
                dtype=infinicore.float32,
                device=infinicore.device(device_str),
            )
        )
        self.b = infinicore.nn.Parameter(
            infinicore.empty(
                (1, 2, 3),
                dtype=infinicore.float32,
                device=infinicore.device(device_str),
            )
        )

    def forward(self):
        return infinicore.add(self.a, self.b)


infinicore_model_infer = InfiniCoreNet()
# ============================================================
# 2. 加载权重
# ============================================================

params_dict = {
    "a": infinicore.empty(
        (1, 2, 3), dtype=infinicore.float32, device=infinicore.device(device_str, 0)
    ),
    "b": infinicore.empty(
        (1, 2, 3), dtype=infinicore.float32, device=infinicore.device(device_str, 0)
    ),
}
infinicore_model_infer.load_state_dict(params_dict)

# ============================================================
# 3. 计算
# ============================================================
infinicore_model_out = infinicore_model_infer()
ref_out = infinicore.add(params_dict["a"], params_dict["b"])


# ============================================================
# 4. 对比结果
# ============================================================
print("InfiniCoreModule 与 Torch (CPU) 最大误差: 手动查看 ")
infinicore_model_out.debug()
ref_out.debug()


# ============================================================
# 5. to测试 - 测试模型在不同设备间的转换
# ============================================================
print("\n" + "=" * 60)
print("5. to测试 - 设备转换测试")
print("=" * 60)


def print_model_state(model, title="状态"):
    """打印模型的参数状态"""
    print(f"\n{title}:")
    print("-" * 40)
    print("Parameters:")
    for name, param in model.named_parameters():
        print(
            f"  {name}: shape={param.shape}, dtype={param.dtype}, device={param.device}"
        )


def verify_device_conversion(model, target_device, use_type_check=False):
    """验证模型参数的设备转换"""
    print("转换后的Parameters:")
    for name, param in model.named_parameters():
        print(
            f"  {name}: shape={param.shape}, dtype={param.dtype}, device={param.device}"
        )
        if use_type_check:
            # 当使用字符串参数时，只检查设备类型
            expected_type = (
                target_device if isinstance(target_device, str) else target_device.type
            )
            assert param.device.type == expected_type, (
                f"参数 {name} 的设备转换失败: 期望类型 {expected_type}, 实际 {param.device.type}"
            )
        else:
            # 使用device对象时，进行完整比较
            assert param.device == target_device, (
                f"参数 {name} 的设备转换失败: 期望 {target_device}, 实际 {param.device}"
            )


# 5.1 打印初始状态
print_model_state(infinicore_model_infer, "5.1 初始状态")

# 定义设备转换测试用例列表
device_conversion_cases = [
    {
        "name": "5.2 转换到CUDA设备",
        "description": "使用 infinicore.device('cuda', 0)",
        "target": infinicore.device("cuda", 0),
        "use_type_check": False,
        "success_msg": "✓ CUDA设备转换验证通过",
    },
    {
        "name": "5.3 转换到CPU设备",
        "description": "使用 infinicore.device('cpu', 0)",
        "target": infinicore.device("cpu", 0),
        "use_type_check": False,
        "success_msg": "✓ CPU设备转换验证通过",
    },
    {
        "name": "5.4 转换到CUDA设备",
        "description": "使用字符串 'cuda'",
        "target": "cuda",
        "use_type_check": True,
        "success_msg": "✓ 字符串参数设备转换验证通过",
    },
]

# 循环测试每个设备转换用例
for case in device_conversion_cases:
    print(f"\n{case['name']} ({case['description']}):")
    print("-" * 40)
    infinicore_model_infer.to(case["target"])
    verify_device_conversion(
        infinicore_model_infer, case["target"], use_type_check=case["use_type_check"]
    )
    print(case["success_msg"])

# 5.5 验证to方法返回self（链式调用支持）
print("\n5.5 测试to方法的返回值（链式调用）:")
print("-" * 40)
result = infinicore_model_infer.to(infinicore.device("cpu", 0))
assert result is infinicore_model_infer, "to方法应该返回self以支持链式调用"
print("✓ to方法返回值验证通过")

print("\n" + "=" * 60)
print("所有to测试通过！")
print("=" * 60 + "\n")
