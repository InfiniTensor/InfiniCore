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
# 5. to测试，buffer测试
# ============================================================
# 等待添加
