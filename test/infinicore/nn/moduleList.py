import os

# ============================================================
# 0. infinicore 包导入，配置测试用 safetensors 临时存储路径
# ============================================================
import sys

import safetensors
import safetensors.torch
import torch
import torch.nn as nn

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python/infinicore"))
)

# 使用临时目录，如果不存在则自动创建
save_dir = os.path.join(os.path.dirname(__file__), "../../tmp")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "torch_modulelist_with_param.safetensors")


def test():
    # ============================================================
    # 1. 使用 PyTorch 定义并保存模型（使用 torch.nn.ModuleList）
    # ============================================================
    class TorchModuleListNet(nn.Module):
        def __init__(self, in_ch=3, hidden_ch=8, out_ch=3):
            super().__init__()
            # 使用 torch.nn.ModuleList
            self.layers = nn.ModuleList(
                [
                    nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_ch),
                    nn.ReLU(),
                    nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_ch),
                    nn.ReLU(),
                    nn.Conv2d(hidden_ch, out_ch, kernel_size=1),
                ]
            )

            # 自定义 Parameter
            self.scale = nn.Parameter(torch.ones(1) * 0.5)
            self.register_buffer("offset", torch.tensor(0.1))

        def forward(self, x):
            # 遍历 ModuleList 中的所有层
            for layer in self.layers:
                x = layer(x)
            # 应用自定义参数和 buffer
            x = x * self.scale + self.offset
            return x

    # ===== 保存 Torch 模型 =====
    torch_model = TorchModuleListNet()
    torch_state_dict = torch_model.state_dict()
    safetensors.torch.save_file(torch_state_dict, save_path)
    print("✓ PyTorch 模型已保存")

    # ============================================================
    # 2. 使用 torch 方式加载并推理
    # ============================================================

    torch_model_infer = TorchModuleListNet()
    torch_model_infer.load_state_dict(safetensors.torch.load_file(save_path))
    torch_model_infer.eval()

    input = torch.rand(1, 3, 8, 8)
    torch_model_out = torch_model_infer(input)
    print("✓ Torch 输出：", torch_model_out.detach().numpy().mean())

    # ============================================================
    # 3. 使用 ModuleList 加载并推理
    # ============================================================

    from nn.modules import Module, ModuleList

    class InfiniCoreModuleListNet(Module):
        def __init__(self, in_ch=3, hidden_ch=8, out_ch=3):
            super().__init__()
            # 使用 ModuleList
            self.layers = ModuleList(
                [
                    nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_ch),
                    nn.ReLU(),
                    nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_ch),
                    nn.ReLU(),
                    nn.Conv2d(hidden_ch, out_ch, kernel_size=1),
                ]
            )

            # 保持与 Torch 模型一致的自定义参数和 buffer
            self.scale = nn.Parameter(torch.ones(1) * 0.5)
            self.register_buffer("offset", torch.tensor(0.1))

        def forward(self, x):
            # 遍历 ModuleList 中的所有层
            for layer in self.layers:
                x = layer(x)
            x = x * self.scale + self.offset
            return x

    # ===== 使用 ModuleListNet 读取 safetensors 并推理 =====
    infinicore_model_infer = InfiniCoreModuleListNet()
    infinicore_model_infer.load_state_dict(safetensors.torch.load_file(save_path))
    infinicore_model_infer.eval()

    infinicore_model_out = infinicore_model_infer.forward(input)
    print("✓ InfiniCore 输出：", infinicore_model_out.detach().numpy().mean())

    # ============================================================
    # 4. 对比结果
    # ============================================================

    diff = (infinicore_model_out - torch_model_out).abs().max().item()
    print(f"✓ ModuleList 与 Torch 最大误差: {diff:.8f}")
    if diff < 1e-9:
        print("✓ ModuleList 与 Torch 精度一致.")
    else:
        print("✗ ModuleList 与 Torch 精度存在差异.")

    # ============================================================
    # 5. 测试 ModuleList 的基本功能
    # ============================================================

    print("\n=== 测试 ModuleList 基本功能 ===")

    # 测试 1: 创建和访问
    module_list = ModuleList([nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)])

    print(f"✓ 创建 ModuleList，长度: {len(module_list)}")
    print(f"✓ 访问第一个模块: {type(module_list[0]).__name__}")
    print(f"✓ 访问第二个模块: {type(module_list[1]).__name__}")

    # 测试 2: append
    module_list.append(nn.Softmax(dim=-1))
    print(f"✓ append 后长度: {len(module_list)}")

    # 测试 3: extend
    module_list.extend([nn.Dropout(0.1), nn.Linear(5, 1)])
    print(f"✓ extend 后长度: {len(module_list)}")

    # 测试 4: 迭代
    print("✓ 迭代 ModuleList:")
    for i, module in enumerate(module_list):
        print(f"  [{i}] {type(module).__name__}")

    # 测试 5: 索引访问
    print(f"✓ 索引访问 module_list[0]: {type(module_list[0]).__name__}")

    # 测试 6: state_dict
    state_dict = module_list.state_dict()
    print(f"✓ state_dict 键数量: {len(state_dict)}")
    print(f"✓ state_dict 包含模块参数: {any('0.' in k for k in state_dict.keys())}")

    # 测试 7: 使用 ModuleList 的模型
    class TestNet(Module):
        def __init__(self):
            super().__init__()
            self.layers = ModuleList([nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    test_model = TestNet()
    test_input = torch.randn(2, 10)
    test_output = test_model.forward(test_input)
    print(f"✓ TestNet 输入形状: {test_input.shape}, 输出形状: {test_output.shape}")

    # 测试 8: __add__ 方法
    ml1 = ModuleList([nn.Linear(10, 5), nn.ReLU()])
    ml2 = ModuleList([nn.Linear(5, 3), nn.Sigmoid()])
    ml3 = ml1 + ml2
    print(f"✓ __add__ 方法测试: {len(ml1)} + {len(ml2)} = {len(ml3)}")
    assert len(ml3) == 4, "合并后的长度应该为 4"

    # 测试 9: pop 方法
    ml4 = ModuleList([nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3)])
    popped = ml4.pop()
    print(
        f"✓ pop 方法测试: 弹出后长度 {len(ml4)}, 弹出模块类型 {type(popped).__name__}"
    )
    assert len(ml4) == 2, "pop 后长度应该为 2"
    assert isinstance(popped, nn.Linear), "弹出的应该是 Linear 模块"

    # 测试 10: __repr__ 方法
    ml5 = ModuleList([nn.Linear(10, 5), nn.ReLU()])
    repr_str = repr(ml5)
    print(f"✓ __repr__ 方法测试: 输出包含类名和模块信息")
    assert "ModuleList" in repr_str or "InfiniCoreModuleList" in repr_str, (
        "repr 应该包含类名"
    )
    assert "Linear" in repr_str, "repr 应该包含模块信息"
    print(repr_str)

    print("\n=== 所有测试通过! ===")

    # ============================================================
    # 6. 前向传播集成测试（参考 infinicore_nn_test.py）
    # ============================================================

    print("\n=== 前向传播集成测试 ===")

    # 使用 ModuleList 创建一个简单的模型
    class TorchModuleListModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)]
            )
            self.scale = nn.Parameter(torch.ones(1) * 0.5)
            self.register_buffer("offset", torch.tensor(0.1))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            x = x * self.scale + self.offset
            return x

    class InfiniCoreModuleListModel(Module):
        def __init__(self):
            super().__init__()
            self.layers = ModuleList([nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)])
            self.scale = nn.Parameter(torch.ones(1) * 0.5)
            self.register_buffer("offset", torch.tensor(0.1))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            x = x * self.scale + self.offset
            return x

    # 创建模型
    torch_model_forward = TorchModuleListModel()
    infinicore_model_forward = InfiniCoreModuleListModel()

    # 复制权重（确保初始权重一致）
    infinicore_model_forward.load_state_dict(
        torch_model_forward.state_dict(), strict=False
    )

    # 设置为评估模式
    torch_model_forward.eval()
    infinicore_model_forward.eval()

    # 创建测试输入
    test_input = torch.randn(2, 10)

    # 前向传播
    with torch.no_grad():
        torch_output = torch_model_forward(test_input)
        infinicore_output = infinicore_model_forward.forward(test_input)

    # 对比结果
    diff = (infinicore_output - torch_output).abs().max().item()
    print(f"✓ 前向传播测试 - 输入形状: {test_input.shape}")
    print(
        f"✓ Torch 输出形状: {torch_output.shape}, 均值: {torch_output.detach().numpy().mean():.8f}"
    )
    print(
        f"✓ InfiniCore 输出形状: {infinicore_output.shape}, 均值: {infinicore_output.detach().numpy().mean():.8f}"
    )
    print(f"✓ 最大误差: {diff:.8f}")

    if diff < 1e-9:
        print("✓ 前向传播集成测试通过：ModuleList 与 Torch ModuleList 结果一致！")
    else:
        print("✗ 前向传播集成测试失败：存在差异")

    # ============================================================
    # 7. 混合模块兼容性测试（PyTorch + InfiniCore 模块混合使用）
    # ============================================================

    print("\n=== 混合模块兼容性测试 ===")

    # 创建一个自定义的 InfiniCore 模块
    class CustomLinear(Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.randn(out_features))

        def forward(self, x):
            return x @ self.weight.t() + self.bias

    # 创建混合 ModuleList（包含 PyTorch 模块和 InfiniCore 模块）
    mixed_list = ModuleList(
        [
            nn.Linear(10, 5),  # PyTorch 模块
            CustomLinear(5, 3),  # 自定义 InfiniCore 模块
            nn.ReLU(),  # PyTorch 模块
        ]
    )

    print(f"✓ 创建混合 ModuleList，长度: {len(mixed_list)}")
    print(f"✓ 模块类型: {[type(m).__name__ for m in mixed_list]}")

    # 测试参数注册
    param_count = sum(1 for _ in mixed_list.parameters())
    print(f"✓ 参数数量: {param_count}")
    assert param_count == 4, (
        f"参数数量应该为 4 (Linear: weight+bias, CustomLinear: weight+bias), 实际为 {param_count}"
    )

    # 测试 state_dict
    mixed_state_dict = mixed_list.state_dict()
    print(f"✓ state_dict 键数量: {len(mixed_state_dict)}")
    assert len(mixed_state_dict) >= 4, "state_dict 应该包含至少 4 个参数"

    # 测试前向传播
    test_input_mixed = torch.randn(2, 10)
    with torch.no_grad():
        x = test_input_mixed
        for module in mixed_list:
            x = module.forward(x)
    print(f"✓ 混合模块前向传播成功，输出形状: {x.shape}")

    print("✓ 混合模块兼容性测试通过！")
