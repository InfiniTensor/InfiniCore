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


def silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU激活函数: x * sigmoid(x)
    """
    sigmoid = 1 / (1 + np.exp(-x))
    return x * sigmoid


def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """
    生成指定形状和数据类型的随机张量
    """
    return np.random.randn(*shape).astype(dtype)


class SILUTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: np.ndarray,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        output: np.ndarray,
        shape_output: List[int] | None,
        stride_output: List[int] | None,
    ):
        super().__init__("silu")
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.output = output
        self.shape_output = shape_output
        self.stride_output = stride_output

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # 添加形状信息
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_output is not None:
            test_writer.add_array(
                test_writer.gguf_key("output.shape"), self.shape_output
            )

        # 添加步幅信息
        if self.stride_input is not None:
            test_writer.add_array(
                test_writer.gguf_key("input.strides"), gguf_strides(*self.stride_input)
            )
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(
                *self.stride_output
                if self.stride_output is not None
                else contiguous_gguf_strides(self.shape_output)
            ),
        )

        # 添加张量数据
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            self.input,
            raw_dtype=np_dtype_to_ggml(self.input.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            self.output,
            raw_dtype=np_dtype_to_ggml(self.output.dtype),
        )

        # 计算并添加预期结果
        ans = silu(self.input.astype(np.float64))
        # 利用广播机制确保ans的shape与input一致
        zero = np.zeros(np.array(self.shape_input), dtype=np.float64)
        ans = ans + zero
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
        # shape, x_stride, y_stride
        ((10,), None, None),
        ((5, 10), None, None),
        ((2, 3, 4), None, None),
        ((100,), (2,), None),
        ((16, 16), (16, 1), (1, 16)),
        ((1, 1024), None, None),
        ((32, 128), None, None),
        ((8, 8, 8, 8), None, None),
        ((256,), (0,), None),
    ]

    # 生成测试用例
    for shape, stride_input, stride_output in _TEST_CASES_:
        # 创建输入张量
        input = random_tensor(shape, dtype)
        input = process_zero_stride_tensor(input, stride_input)

        # 创建输出占位张量
        output = np.empty(tuple(0 for _ in shape), dtype=dtype)

        # 添加测试用例
        test_cases.append(
            SILUTestCase(
                input=input,
                output=output,
                shape_input=shape,
                stride_input=stride_input,
                shape_output=shape,
                stride_output=stride_output,
            )
        )

    # 添加所有测试用例并保存
    test_writer.add_tests(test_cases)
    test_writer.save()


if __name__ == "__main__":
    _TENSOR_DTYPES_ = [np.float32, np.float16, bfloat16]
    dtype_filename_map = {
        np.float32: "silu_f32.gguf",
        np.float16: "silu_f16.gguf",
        bfloat16: "silu_bf16.gguf",
    }

    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)
