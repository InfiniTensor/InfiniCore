import math
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


def gelu(input: np.ndarray, approximate: Optional[str] = None) -> np.ndarray:
    """
    高斯误差线性单元(GELU)激活函数

    参数:
        input (np.ndarray): 输入张量
        approximate (str): 近似模式，'none'或'tanh'

    返回:
        np.ndarray: GELU激活后的输出

    根据approximate参数选择不同的计算方法:
    - 当 approximate = 'none' 时: GELU(x) = x * Φ(x)
      其中Φ(x)是标准正态分布的累积分布函数
    - 当 approximate = 'tanh' 时:
      GELU(x) = 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    x = input

    if approximate is None:
        # 使用误差函数erf计算高斯CDF
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
        return x * cdf

    elif approximate == "tanh":
        # 使用tanh近似公式
        inner = np.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + np.tanh(inner))

    else:
        raise ValueError(
            f"Unsupported approximate mode: '{approximate}'. "
            "Supported modes are 'none' and 'tanh'."
        )


def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """
    生成指定形状和数据类型的随机张量
    """
    return np.random.randn(*shape).astype(dtype)


class GeluTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: np.ndarray,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        output: np.ndarray,
        shape_output: List[int] | None,
        stride_output: List[int] | None,
        approximate: Optional[str] = None,
    ):
        super().__init__("gelu")
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.output = output
        self.shape_output = shape_output
        self.stride_output = stride_output
        self.approximate = approximate

    def write_test(self, test_writer: InfiniopTestWriter):
        super().write_test(test_writer)

        # 添加形状信息
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_output is not None:
            test_writer.add_array(
                test_writer.gguf_key("output.shape"), self.shape_output
            )

        # 添加步长信息
        if self.stride_input is not None:
            test_writer.add_array(
                test_writer.gguf_key("input.strides"), gguf_strides(*self.stride_input)
            )
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(
                *(
                    self.stride_output
                    if self.stride_output is not None
                    else contiguous_gguf_strides(self.shape_output)
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
            test_writer.gguf_key("output"),
            self.output,
            raw_dtype=np_dtype_to_ggml(self.output.dtype),
        )

        # 计算并添加预期结果
        ans = gelu(self.input.astype(np.float64))
        # 利用广播机制确保ans的shape与input一致
        zero = np.zeros(np.array(self.shape_input), dtype=np.float64)
        ans = ans + zero
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


def gen_gguf(dtype: np.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    _TEST_CASES_ = [
        # (shape, stride_input, stride_output)
        ((256,), None, None),
        ((16, 512), None, None),
        ((4, 4, 512), None, None),
        ((2, 3, 4, 5), None, None),
        ((1,), None, None),
        ((1, 1, 1), None, None),
        ((13, 4), (10, 1), None),
        ((16, 16), (32, 1), None),
        ((3, 4, 5), (25, 5, 1), None),
        ((2, 3, 4, 5), (65, 20, 5, 1), None),
        ((5, 8), (1, 0), None),
        ((4, 5, 6), (10, 0, 1), None),
        ((4, 4, 512), None, (2100, 512, 1)),
    ]

    # 生成测试用例
    for shape, stride_input, stride_output in _TEST_CASES_:
        # 生成随机张量
        input = random_tensor(shape, dtype)
        # 处理零步长情况
        input = process_zero_stride_tensor(input, stride_input)
        # 创建输出张量（初始为空）
        output = np.empty(tuple(0 for _ in shape), dtype=dtype)
        # 创建测试用例
        test_case = GeluTestCase(
            input=input,
            shape_input=shape,
            stride_input=stride_input,
            output=output,
            shape_output=shape,
            stride_output=stride_output,
            approximate="tanh",
        )
        test_cases.append(test_case)

    # 添加所有测试用例并保存
    test_writer.add_tests(test_cases)
    test_writer.save()


if __name__ == "__main__":
    _TENSOR_DTYPES_ = [np.float32, np.float16, bfloat16]
    dtype_filename_map = {
        np.float32: "gelu_f32.gguf",
        np.float16: "gelu_f16.gguf",
        bfloat16: "gelu_bf16.gguf",
    }

    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)
