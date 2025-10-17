import safetensors.torch
import torch
import torch.nn as nn
import safetensors

# ============================================================
# 1. 使用 PyTorch 定义并保存模型
# ============================================================

class TorchConvNet(nn.Module):
    def __init__(self, in_ch=3, hidden_ch=8, out_ch=3):
        super().__init__()
        # 主体网络
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU()

        # 自定义 Parameter（例如一个可学习缩放因子）
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

        # 注册一个 buffer（非参数，例如推理时的固定偏移）
        self.register_buffer("offset", torch.tensor(0.1))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        # 应用自定义参数和 buffer
        x = x * self.scale + self.offset
        return x


# ===== 保存 Torch 模型 =====
torch_model = TorchConvNet()
torch_state_dict = torch_model.state_dict()
safetensors.torch.save_file(torch_state_dict, "torch_convnet_with_param.safetensors")

# ============================================================
# 2. 使用 torch 方式加载并推理
# ============================================================

torch_model_infer = TorchConvNet()
torch_model_infer.load_state_dict(safetensors.torch.load_file("torch_convnet_with_param.safetensors"))
torch_model_infer.eval()

input = torch.rand(1, 3, 8, 8)
torch_model_out = torch_model_infer(input)
print("Torch 输出：", torch_model_out.detach().numpy().mean())

# ============================================================
# 3. 使用 InfiniCore.Module 系统加载并推理
# ============================================================

from python.infinicore.nn.modules import InfiniCoreModule

class InfiniCoreConvNet(InfiniCoreModule):
    def __init__(self, in_ch=3, hidden_ch=8, out_ch=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU()

        # 保持与 Torch 模型一致的自定义参数和 buffer
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
        self.register_buffer("offset", torch.tensor(0.1))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = x * self.scale + self.offset
        return x

# ===== 使用 InfiniCoreConvNet 读取 safetensors 并推理 =====
infinicore_model_infer = InfiniCoreConvNet()
infinicore_model_infer.load_state_dict(safetensors.torch.load_file("torch_convnet_with_param.safetensors"))
infinicore_model_infer.eval()

infinicore_model_out = infinicore_model_infer.forward(input)
print("InfiniCore 输出：", infinicore_model_out.detach().numpy().mean())

# ============================================================
# 4. 对比结果
# ============================================================

diff = (infinicore_model_out - torch_model_out).abs().max().item()
print(f"InfiniCoreModule 与 Torch 最大误差: {diff:.8f}")
if diff < 1e-6:
    print("✅ InfiniCoreModule 与 Torch 精度一致!")
else:
    print("❌ InfiniCoreModule 与 Torch 精度存在差异!")
