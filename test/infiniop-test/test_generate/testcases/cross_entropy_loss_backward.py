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


def cross_entropy_backward(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Step 1: 重塑为二维张量 (N*S, C)，S=空间维度大小
    orig_shape = logits.shape
    num_classes = logits.shape[-1]
    logits_2d = logits.reshape(-1, num_classes)

    # Step 2: 计算softmax概率
    exp_logits = np.exp(logits_2d - np.max(logits_2d, axis=1, keepdims=True))
    probs_2d = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (N*S, C)

    # Step 3: 处理target（整数标签或one-hot）
    if len(target.shape) == len(orig_shape) - 1:
        target_flat = target.reshape(-1)  # 展平为 (N*S,)
        one_hot = np.zeros_like(probs_2d)
        one_hot[np.arange(len(target_flat)), target_flat] = 1
    else:  # one-hot标签
        one_hot = target.reshape(-1, num_classes)  # 展平为 (N*S, C)

    # Step 4: 计算梯度 (p_i - y_i) / 总样本数（含空间维度）
    grad_2d = (probs_2d - one_hot) / logits_2d.shape[0]

    # Step 5: 恢复原始形状
    grad_logits = grad_2d.reshape(orig_shape)
    return grad_logits


class CrossEntropyLossBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        logits: np.ndarray,
        shape_logits: List[int] | None,
        stride_logits: List[int] | None,
        target: np.ndarray,
        shape_target: List[int] | None,
        stride_target: List[int] | None,
        grad_logits: np.ndarray,
        shape_grad_logits: List[int] | None,
        stride_grad_logits: List[int] | None,
    ):
        super().__init__("cross_entropy_loss_backward")
        self.logits = logits
        self.shape_logits = shape_logits
        self.stride_logits = stride_logits
        self.target = target
        self.shape_target = shape_target
        self.stride_target = stride_target
        self.grad_logits = grad_logits
        self.shape_grad_logits = shape_grad_logits
        self.stride_grad_logits = stride_grad_logits

    def write_test(self, test_writer: InfiniopTestWriter):
        super().write_test(test_writer)

        # 添加形状信息
        if self.shape_logits is not None:
            test_writer.add_array(
                test_writer.gguf_key("logits.shape"), self.shape_logits
            )
        if self.shape_target is not None:
            test_writer.add_array(
                test_writer.gguf_key("target.shape"), self.shape_target
            )
        if self.shape_grad_logits is not None:
            test_writer.add_array(
                test_writer.gguf_key("grad_logits.shape"), self.shape_grad_logits
            )

        # 添加步长信息
        if self.stride_logits is not None:
            test_writer.add_array(
                test_writer.gguf_key("logits.strides"),
                gguf_strides(*self.stride_logits),
            )
        if self.stride_target is not None:
            test_writer.add_array(
                test_writer.gguf_key("target.strides"),
                gguf_strides(*self.stride_logits),
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
            test_writer.gguf_key("logits"),
            self.logits,
            raw_dtype=np_dtype_to_ggml(self.logits.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("target"),
            self.target,
            raw_dtype=np_dtype_to_ggml(self.target.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_logits"),
            self.grad_logits,
            raw_dtype=np_dtype_to_ggml(self.target.dtype),
        )

        # 计算参考结果（使用float64精度）
        logits_f64 = self.logits.astype(np.float64)
        target_i32 = self.target.astype(np.int32)
        grad_logits = cross_entropy_backward(logits_f64, target_i32)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            grad_logits,
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("cross_entropy_loss_backward.gguf")
    test_cases = []

    # 测试用例配置
    _TEST_CASES_ = [
        # logits_shape, target_shape, logits_strides, target_strides, grad_logits_strides
        ((8, 5), (8,), None, None, None),
        ((8, 5), (8, 5), None, None, None),
        ((16, 1), (16,), (1, 1), (1,), (1, 1)),
        ((16, 2), (16, 2), (2, 1), (2, 1), (2, 1)),
        ((64, 1000), (64,), None, None, None),
        (
            (64, 1000),
            (64, 1000),
            (8000, 8),
            (8000, 8),
            (8000, 8),
        ),
        (
            (10, 20),
            (10,),
            (40, 2),
            (10,),
            (40, 2),
        ),
        (
            (5, 3),
            (5, 3),
            (6, 2),
            (15, 5),
            (6, 2),
        ),
        ((1, 10), (1,), None, None, None),
        (
            (8, 1),
            (8,),
            (0, 1),
            (0,),
            (0, 1),
        ),
        (
            (4, 32, 32, 10),
            (4, 32, 32),
            (10240, 320, 10, 1),
            (1024, 32, 1),
            (10240, 320, 10, 1),
        ),
    ]

    _TENSOR_DTYPES_ = [np.float32, np.float16, bfloat16]
    for dtype in _TENSOR_DTYPES_:
        for (
            shape_logits,
            shape_target,
            stride_logits,
            stride_target,
            stride_grad_logits,
        ) in _TEST_CASES_:
            # 生成随机张量
            logits = np.random.randn(*shape_logits).astype(dtype)
            target = np.random.randint(
                low=0, high=shape_logits[-1], size=shape_target
            ).astype(dtype)

            # 处理零步长情况
            logits = process_zero_stride_tensor(logits, stride_logits)
            target = process_zero_stride_tensor(target, stride_target)

            # 创建输出张量（初始为空）
            grad_logits = np.empty(tuple(0 for _ in shape_logits), dtype=dtype)

            # 创建测试用例
            test_case = CrossEntropyLossBackwardTestCase(
                logits=logits,
                shape_logits=shape_logits,
                stride_logits=stride_logits,
                target=target,
                shape_target=shape_target,
                stride_target=stride_target,
                grad_logits=grad_logits,
                shape_grad_logits=shape_logits,
                stride_grad_logits=stride_grad_logits,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
