from typing import List

import gguf
import numpy as np
from ml_dtypes import bfloat16

from .. import (
    InfiniopTestCase,
    InfiniopTestWriter,
    contiguous_gguf_strides,
    gguf_strides,
    np_dtype_to_ggml,
    process_zero_stride_tensor,
)


def generate_one_hot(shape: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """生成指定形状的 one-hot 数组"""
    num_classes = shape[-1]  # 获取类别数 C
    # 生成类别索引：形状为 probs 去掉最后一个维度的形状
    indices = np.random.randint(low=0, high=num_classes, size=shape[:-1])
    # 通过单位矩阵索引生成 one-hot 数组
    return np.eye(num_classes, dtype=dtype)[indices]


def cross_entropy_backward(probs: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Step 1: 重塑为二维张量 (N*S, C)，S=空间维度大小
    orig_shape = probs.shape
    num_classes = probs.shape[-1]
    probs_2d = probs.reshape(-1, num_classes)
    target_2d = target.reshape(-1, num_classes)

    # Step 2: 计算梯度 (p_i - y_i) / 总样本数（含空间维度）
    grad_2d = (probs_2d - target_2d) / probs_2d.shape[0]

    # Step 3: 恢复原始形状
    grad_logits = grad_2d.reshape(orig_shape)
    return grad_logits


class CrossEntropyLossBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        probs: np.ndarray,
        shape_probs: List[int] | None,
        stride_probs: List[int] | None,
        target: np.ndarray,
        shape_target: List[int] | None,
        stride_target: List[int] | None,
        grad_logits: np.ndarray,
        shape_grad_logits: List[int] | None,
        stride_grad_logits: List[int] | None,
    ):
        super().__init__("cross_entropy_loss_backward")
        self.probs = probs
        self.shape_probs = shape_probs
        self.stride_probs = stride_probs
        self.target = target
        self.shape_target = shape_target
        self.stride_target = stride_target
        self.grad_logits = grad_logits
        self.shape_grad_logits = shape_grad_logits
        self.stride_grad_logits = stride_grad_logits

    def write_test(self, test_writer: InfiniopTestWriter):
        super().write_test(test_writer)

        # 添加形状信息
        if self.shape_probs is not None:
            test_writer.add_array(test_writer.gguf_key("probs.shape"), self.shape_probs)
        if self.shape_target is not None:
            test_writer.add_array(
                test_writer.gguf_key("target.shape"), self.shape_target
            )
        if self.shape_grad_logits is not None:
            test_writer.add_array(
                test_writer.gguf_key("grad_logits.shape"), self.shape_grad_logits
            )

        # 添加步长信息
        if self.stride_probs is not None:
            test_writer.add_array(
                test_writer.gguf_key("probs.strides"),
                gguf_strides(*self.stride_probs),
            )
        if self.stride_target is not None:
            test_writer.add_array(
                test_writer.gguf_key("target.strides"),
                gguf_strides(*self.stride_target),
            )
        test_writer.add_array(
            test_writer.gguf_key("grad_logits.strides"),
            gguf_strides(
                *self.stride_grad_logits
                if self.stride_grad_logits is not None
                else contiguous_gguf_strides(self.shape_grad_logits)
            ),
        )

        # 添加张量数据
        test_writer.add_tensor(
            test_writer.gguf_key("probs"),
            self.probs,
            raw_dtype=np_dtype_to_ggml(self.probs.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("target"),
            self.target,
            raw_dtype=np_dtype_to_ggml(self.target.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_logits"),
            self.grad_logits,
            raw_dtype=np_dtype_to_ggml(self.grad_logits.dtype),
        )
        # 计算参考结果（使用float64精度）
        zero = np.zeros(np.array(self.shape_probs), dtype=np.float64)
        probs_f64 = self.probs.astype(np.float64) + zero
        target_i32 = self.target.astype(np.int32) + zero
        ans = cross_entropy_backward(probs_f64, target_i32)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans,
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )


def gen_gguf(dtype: np.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    # 测试用例配置
    _TEST_CASES_ = [
        # probs_shape, target_shape, logits_strides, target_strides, grad_logits_strides
        ((8, 5), (8, 5), None, None, None),
        ((1, 10), (1, 10), None, None, None),
        ((64, 1000), (64, 1000), None, None, None),
        (
            (16, 8),
            (16, 8),
            (10, 1),
            (10, 1),
            (10, 1),
        ),
        (
            (4, 10, 32, 32),
            (4, 10, 32, 32),
            (10240, 1024, 32, 1),
            (10240, 1024, 32, 1),
            (10240, 1024, 32, 1),
        ),
        (
            (5, 3),
            (5, 3),
            (6, 2),
            (15, 5),
            (6, 2),
        ),
        (
            (8, 1),
            (8, 1),
            (0, 1),
            (1, 1),
            (1, 1),
        ),
        (
            (32, 20, 50),
            (32, 20, 50),
            (1000, 50, 1),
            (1000, 50, 1),
            (1000, 50, 1),
        ),
        ((10, 2), (10, 2), None, None, None),
        (
            (2, 256, 256, 20),
            (2, 256, 256, 20),
            (1310720, 5120, 20, 1),
            (1310720, 5120, 20, 1),
            (1310720, 5120, 20, 1),
        ),
        (
            (12, 7),
            (12, 7),
            (14, 2),
            (21, 3),
            (14, 2),
        ),
        ((6, 1), (6, 1), None, None, None),
        (
            (2, 8, 64, 64, 10),
            (2, 8, 64, 64, 10),
            (327680, 40960, 640, 10, 1),
            (327680, 40960, 640, 10, 1),
            (327680, 40960, 640, 10, 1),
        ),
    ]

    for (
        shape_probs,
        shape_target,
        stride_probs,
        stride_target,
        stride_grad_logits,
    ) in _TEST_CASES_:
        # 生成随机张量
        probs = np.random.randn(*shape_probs).astype(dtype)
        target = generate_one_hot(shape_target, dtype=dtype)
        # 处理零步长情况
        probs = process_zero_stride_tensor(probs, stride_probs)
        target = process_zero_stride_tensor(target, stride_target)
        # 创建输出张量（初始为空）
        grad_logits = np.empty(tuple(0 for _ in shape_probs), dtype=dtype)
        # 创建测试用例
        test_case = CrossEntropyLossBackwardTestCase(
            probs=probs,
            shape_probs=shape_probs,
            stride_probs=stride_probs,
            target=target,
            shape_target=shape_target,
            stride_target=stride_target,
            grad_logits=grad_logits,
            shape_grad_logits=shape_probs,
            stride_grad_logits=stride_grad_logits,
        )
        test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()


if __name__ == "__main__":
    _TENSOR_DTYPES_ = [np.float32, np.float16, bfloat16]
    dtype_filename_map = {
        np.float32: "cross_entropy_loss_backward_f32.gguf",
        np.float16: "cross_entropy_loss_backward_f16.gguf",
        bfloat16: "cross_entropy_loss_backward_bf16.gguf",
    }

    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)
