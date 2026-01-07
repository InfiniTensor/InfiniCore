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

# 5.1 打印初始状态
print("\n5.1 初始状态:")
print("-" * 40)
print("Parameters:")
for name, param in infinicore_model_infer.named_parameters():
    print(f"  {name}: shape={param.shape}, dtype={param.dtype}, device={param.device}")
print("Buffers:")
buffers_exist = False
for name, buf in infinicore_model_infer.named_buffers():
    buffers_exist = True
    print(f"  {name}: shape={buf.shape}, dtype={buf.dtype}, device={buf.device}")
if not buffers_exist:
    print("  (无buffers)")

# 5.2 测试转换到CUDA设备（使用device对象）
print("\n5.2 转换到CUDA设备 (使用 infinicore.device('cuda', 0)):")
print("-" * 40)
target_device_cuda = infinicore.device("cuda", 0)
infinicore_model_infer.to(target_device_cuda)

print("转换后的Parameters:")
for name, param in infinicore_model_infer.named_parameters():
    print(f"  {name}: shape={param.shape}, dtype={param.dtype}, device={param.device}")
    # 验证设备是否正确转换
    assert param.device == target_device_cuda, (
        f"参数 {name} 的设备转换失败: 期望 {target_device_cuda}, 实际 {param.device}"
    )
if buffers_exist:
    print("转换后的Buffers:")
    for name, buf in infinicore_model_infer.named_buffers():
        print(f"  {name}: shape={buf.shape}, dtype={buf.dtype}, device={buf.device}")
        assert buf.device == target_device_cuda, (
            f"Buffer {name} 的设备转换失败: 期望 {target_device_cuda}, 实际 {buf.device}"
        )
print("✓ CUDA设备转换验证通过")

# 5.3 测试转换到CPU设备（使用device对象）
print("\n5.3 转换到CPU设备 (使用 infinicore.device('cpu', 0)):")
print("-" * 40)
target_device_cpu = infinicore.device("cpu", 0)
infinicore_model_infer.to(target_device_cpu)

print("转换后的Parameters:")
for name, param in infinicore_model_infer.named_parameters():
    print(f"  {name}: shape={param.shape}, dtype={param.dtype}, device={param.device}")
    # 验证设备是否正确转换
    assert param.device == target_device_cpu, (
        f"参数 {name} 的设备转换失败: 期望 {target_device_cpu}, 实际 {param.device}"
    )
if buffers_exist:
    print("转换后的Buffers:")
    for name, buf in infinicore_model_infer.named_buffers():
        print(f"  {name}: shape={buf.shape}, dtype={buf.dtype}, device={buf.device}")
        assert buf.device == target_device_cpu, (
            f"Buffer {name} 的设备转换失败: 期望 {target_device_cpu}, 实际 {buf.device}"
        )
print("✓ CPU设备转换验证通过")

# 5.4 测试使用字符串参数转换到CUDA设备
print("\n5.4 转换到CUDA设备 (使用字符串 'cuda'):")
print("-" * 40)
infinicore_model_infer.to("cuda")

print("转换后的Parameters:")
for name, param in infinicore_model_infer.named_parameters():
    print(f"  {name}: shape={param.shape}, dtype={param.dtype}, device={param.device}")
    # 验证设备是否正确转换（字符串'cuda'会被转换为cuda设备）
    assert param.device.type == "cuda", (
        f"参数 {name} 的设备转换失败: 期望 cuda, 实际 {param.device.type}"
    )
if buffers_exist:
    print("转换后的Buffers:")
    for name, buf in infinicore_model_infer.named_buffers():
        print(f"  {name}: shape={buf.shape}, dtype={buf.dtype}, device={buf.device}")
        assert buf.device.type == "cuda", (
            f"Buffer {name} 的设备转换失败: 期望 cuda, 实际 {buf.device.type}"
        )
print("✓ 字符串参数设备转换验证通过")

# 5.5 验证to方法返回self（链式调用支持）
print("\n5.5 测试to方法的返回值（链式调用）:")
print("-" * 40)
result = infinicore_model_infer.to(infinicore.device("cpu", 0))
assert result is infinicore_model_infer, "to方法应该返回self以支持链式调用"
print("✓ to方法返回值验证通过")

print("\n" + "=" * 60)
print("所有to测试通过！")
print("=" * 60 + "\n")
