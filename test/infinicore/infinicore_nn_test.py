import safetensors.torch
import torch
import torch.nn as nn
import safetensors

# ============================================================
# 0. infinicore 包导入，配置测试用 safetensors 临时存储路径
# ============================================================
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python/infinicore')))

save_dir = os.path.join(os.path.dirname(__file__), '../../tmp')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "torch_convnet_with_param.safetensors")

# ============================================================
# 1. 使用 PyTorch 定义并保存模型
# ============================================================
print("===== 开始 CPU 一致性测试 =====")

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

        # 自定义 Parameter
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

        # 注册一个 buffer
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
safetensors.torch.save_file(torch_state_dict, save_path)

# ============================================================
# 2. 使用 torch 方式加载并推理
# ============================================================

torch_model_infer = TorchConvNet()
torch_model_infer.load_state_dict(safetensors.torch.load_file(save_path))
torch_model_infer.eval()

input = torch.rand(1, 3, 8, 8)
torch_model_out = torch_model_infer(input)

# ============================================================
# 3. 使用 infiniCore.nn.module 加载并推理
# ============================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python/infinicore')))

from nn import Module

class InfiniCoreConvNet(Module):
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
infinicore_model_infer.load_state_dict(safetensors.torch.load_file(save_path))
infinicore_model_infer.eval()

infinicore_model_out = infinicore_model_infer.forward(input)

# ============================================================
# 4. 对比结果
# ============================================================

diff_cpu = (infinicore_model_out - torch_model_out).abs().max().item()
print(f"InfiniCoreModule 与 Torch (CPU) 最大误差: {diff_cpu:.6e}")

if diff_cpu < 1e-9:
    print("CPU 模式下 InfiniCore 与 Torch 输出完全一致.")
else:
    print("CPU 模式下输出存在差异.")


# ============================================================
# 5. GPU 一致性测试（可选）
# ============================================================

if torch.cuda.is_available():
    print("\n===== 开始 GPU 一致性测试 =====")

    # 将模型与输入都迁移到 GPU
    torch_model_infer_gpu = TorchConvNet().to("cuda")
    torch_model_infer_gpu.load_state_dict(safetensors.torch.load_file(save_path))
    torch_model_infer_gpu.eval()

    infinicore_model_infer_gpu = InfiniCoreConvNet().to("cuda")
    infinicore_model_infer_gpu.load_state_dict(safetensors.torch.load_file(save_path))
    infinicore_model_infer_gpu.eval()

    # 生成 GPU 输入
    input_gpu = input.to("cuda")

    # 分别前向推理
    torch_out_gpu = torch_model_infer_gpu(input_gpu)
    infinicore_out_gpu = infinicore_model_infer_gpu.forward(input_gpu)

    # 结果比较
    diff_gpu = (infinicore_out_gpu - torch_out_gpu).abs().max().item()
    print(f"InfiniCoreModule 与 Torch (GPU) 最大误差: {diff_gpu:.6e}")

    if diff_gpu < 1e-9:
        print("GPU 模式下 InfiniCore 与 Torch 输出完全一致.")
    else:
        print("GPU 模式下输出存在差异.")
else:
    print("\n 未检测到 GPU，跳过 GPU 一致性测试。")