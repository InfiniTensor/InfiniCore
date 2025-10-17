import safetensors.torch
import torch
import torch.nn as nn
import safetensors

# ============================================================
# 1. 使用 PyTorch 定义并保存模型
# ============================================================

class TorchMLP(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=8, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

torch_model = TorchMLP()
torch_state_dict = torch_model.state_dict()

safetensors.torch.save_file(torch_state_dict, "torch_model.safetensors")

# ============================================================
# 2. 使用 torch 方式加载并推理
# ============================================================

torch_model_infer = TorchMLP()
torch_model_infer.load_state_dict(safetensors.torch.load_file("torch_model.safetensors"))
torch_model_infer.eval()

input = torch.rand(1, 4)
torch_model_out = torch_model_infer(input)
print("Torch 输出：", torch_model_out.detach().numpy())

# ============================================================
# 3. 使用 InfiniCore.Module 系统加载并推理
# ============================================================

# ===== 下面定义一个与 TorchMLP 对应的 InfiniCoreModule类 =====
from python.infinicore.nn.modules import InfiniCoreModule

class InfiniCoreMLP(InfiniCoreModule):
    def __init__(self, in_dim=4, hidden_dim=8, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===== 使用 InfiniCoreMLP 读取 safetensors 并推理 =====
infinicore_model_infer = InfiniCoreMLP()
infinicore_model_infer.load_state_dict(safetensors.torch.load_file("torch_model.safetensors"))
infinicore_model_out = infinicore_model_infer.forward(input)

print("InfiniCore 输出：", infinicore_model_out.detach().numpy())

# ============================================================
# 4. 对比结果
# ============================================================

diff = (infinicore_model_out - torch_model_out).abs().max().item()
print(f"InfiniCoreModule 与 Torch 最大误差: {diff:.6f}")
if diff < 1e-6:
    print("✅ InfiniCoreModule 与 Torch 精度一致!")
else:
    print("❌ InfiniCoreModule 与 Torch 精度存在差异!")
