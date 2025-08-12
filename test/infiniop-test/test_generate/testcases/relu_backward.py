from typing import List, Optional

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


def relu_backward(input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    """ReLU反向算子的参考实现"""
    mask = input > 0
    return mask * grad_output


def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """
    生成指定形状和数据类型的随机张量
    """
    return np.random.randn(*shape).astype(dtype)


class ReluBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: np.ndarray,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        grad_output: np.ndarray,
        shape_grad_output: List[int] | None,
        stride_grad_output: List[int] | None,
        grad_input: np.ndarray,
        shape_grad_input: List[int] | None,
        stride_grad_input: List[int] | None,
    ):
        super().__init__("relu_backward")
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.grad_output = grad_output
        self.shape_grad_output = shape_grad_output
        self.stride_grad_output = stride_grad_output
        self.grad_input = grad_input
        self.shape_grad_input = shape_grad_input
        self.stride_grad_input = stride_grad_input

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # 添加形状信息
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_grad_output is not None:
            test_writer.add_array(
                test_writer.gguf_key("grad_output.shape"), self.shape_grad_output
            )
        if self.shape_grad_input is not None:
            test_writer.add_array(
                test_writer.gguf_key("grad_input.shape"), self.shape_grad_input
            )

        # 添加步长信息
        if self.stride_input is not None:
            test_writer.add_array(
                test_writer.gguf_key("input.strides"), gguf_strides(*self.stride_input)
            )
        if self.stride_grad_output is not None:
            test_writer.add_array(
                test_writer.gguf_key("grad_output.strides"),
                gguf_strides(*self.stride_grad_output),
            )
        test_writer.add_array(
            test_writer.gguf_key("grad_input.strides"),
            gguf_strides(
                *(
                    self.stride_grad_input
                    if self.stride_grad_input is not None
                    else contiguous_gguf_strides(self.shape_grad_input)
                )
            ),
        )

        # 添加张量数据
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            self.input,
            raw_dtype=np_dtype_to_ggml(self.input.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_output"),
            self.grad_output,
            raw_dtype=np_dtype_to_ggml(self.grad_output.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_input"),
            self.grad_input,
            raw_dtype=np_dtype_to_ggml(self.grad_input.dtype),
        )

        # 计算并添加预期结果
        grad_input = relu_backward(
            self.input.astype(np.float64), self.grad_output.astype(np.float64)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            grad_input,
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )


def gen_gguf(dtype: np.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    # 测试用例配置
    _TEST_CASES_ = [
        # (shape, stride_input, stride_grad_output, stride_grad_input)
        ((256,), None, None, None),
        ((16, 512), None, None, None),
        ((4, 4, 512), None, None, None),
        ((2, 3, 4, 5), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4), (13, 1), (13, 1), (13, 1)),
        ((10, 20), (0, 1), None, None),
        ((5, 8), (0, 1), None, None),
        ((3, 15), (1, 0), None, None),
        ((4, 5, 6), (0, 10, 1), None, None),
        ((4, 5, 6), (5, 1, 0), None, None),
        ((1, 1), None, None, None),
    ]

    # 生成测试用例
    for shape, stride_input, stride_grad_output, stride_grad_input in _TEST_CASES_:
        # 生成随机张量
        input = random_tensor(shape, dtype)
        grad_output = random_tensor(shape, dtype)

        # 处理零步长情况
        input = process_zero_stride_tensor(input, stride_input)
        grad_output = process_zero_stride_tensor(grad_output, stride_grad_output)

        # 创建输出张量（初始为空）
        grad_input = np.empty(tuple(0 for _ in shape), dtype=dtype)

        # 创建测试用例
        test_case = ReluBackwardTestCase(
            input=input,
            shape_input=shape,
            stride_input=stride_input,
            grad_output=grad_output,
            shape_grad_output=shape,
            stride_grad_output=stride_grad_output,
            grad_input=grad_input,
            shape_grad_input=shape,
            stride_grad_input=stride_grad_input,
        )
        test_cases.append(test_case)

    # 添加所有测试用例并保存
    test_writer.add_tests(test_cases)
    test_writer.save()


if __name__ == "__main__":
    _TENSOR_DTYPES_ = [np.float32, np.float16, bfloat16]
    dtype_filename_map = {
        np.float32: "relu_backward_f32.gguf",
        np.float16: "relu_backward_f16.gguf",
        bfloat16: "relu_backward_bf16.gguf",
    }

    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)
