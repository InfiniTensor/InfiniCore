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


def gelu_backward(
    input: np.ndarray, grad_output: np.ndarray, approximate: Optional[str] = None
) -> np.ndarray:
    """
    GELU 激活函数的反向传播（梯度计算）

    参数:
        input (np.ndarray): 前向传播的输入
        grad_output (np.ndarray): 上游梯度（即损失函数对 GELU 输出的梯度）
        approximate (str): 近似模式，None 或'tanh'

    返回:
        np.ndarray: 梯度（损失函数对输入的梯度）

    根据 approximate 参数选择不同的梯度计算方法:
    精确模式 (None):
        d_gelu/dx = Φ(x) + x * φ(x)
        其中 φ(x) 是标准正态分布的概率密度函数

    近似模式 ('tanh'):
        d_gelu/dx = 0.5 * (1 + tanh(k))
                    + 0.5 * x * (1 - tanh²(k)) * dk/dx
        其中 k = √(2/π) * (x + 0.044715 * x³)
        且 dk/dx = √(2/π) * (1 + 0.134145 * x²)
    """

    x = input

    if approximate is None:
        # φ(x) = 1/√(2π) * e^(-x²/2)
        phi = (1.0 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * x**2)

        # Φ(x) = (1 + erf(x/√2)) / 2
        erf_vectorized = np.vectorize(math.erf, otypes=[np.float64])
        phi_cumulative = 0.5 * (1.0 + erf_vectorized(input / math.sqrt(2)))

        # d_gelu/dx = Φ(x) + x * φ(x)
        grad = phi_cumulative + input * phi

    elif approximate == "tanh":
        # k = √(2/π) * (x + 0.044715 * x³)
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        k = sqrt_2_over_pi * (input + 0.044715 * input**3)
        tanh_k = np.tanh(k)

        # dk/dx = √(2/π) * (1 + 0.044715 * 3*x²) = √(2/π) * (1 + 0.134145*x²)
        dk_dx = sqrt_2_over_pi * (1.0 + 0.134145 * input**2)

        # d_gelu/dx = 0.5*(1+tanh(k)) + 0.5*x*(1-tanh²(k))*dk/dx
        grad = 0.5 * (1.0 + tanh_k) + 0.5 * input * (1.0 - tanh_k**2) * dk_dx

    else:
        raise ValueError(
            f"Unsupported approximate mode: '{approximate}'. "
            "Supported modes are None and 'tanh'."
        )

    # 乘以上游梯度 (链式法则)
    return grad_output * grad


def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """
    生成指定形状和数据类型的随机张量
    """
    return np.random.randn(*shape).astype(dtype)


class GeluBackwardTestCase(InfiniopTestCase):
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
        approximate_mode: Optional[str] = None,
    ):
        super().__init__("gelu_backward")
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.grad_output = grad_output
        self.shape_grad_output = shape_grad_output
        self.stride_grad_output = stride_grad_output
        self.grad_input = grad_input
        self.shape_grad_input = shape_grad_input
        self.stride_grad_input = stride_grad_input
        self.approximate_mode = approximate_mode

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
        grad_input = gelu_backward(
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
        ((8, 8, 256), None, None, None),
        ((2, 16, 32, 64), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((20, 10), (20, 2), (20, 2), (20, 2)),
        ((7, 9), (1, 0), None, None),
        ((12, 15), (1, 0), None, None),
        ((4, 5, 6), (0, 10, 1), None, None),
        ((1, 1), None, None, None),
        ((1,), None, None, None),
        ((1, 100), None, None, None),
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
        test_case = GeluBackwardTestCase(
            input=input,
            shape_input=shape,
            stride_input=stride_input,
            grad_output=grad_output,
            shape_grad_output=shape,
            stride_grad_output=stride_grad_output,
            grad_input=grad_input,
            shape_grad_input=shape,
            stride_grad_input=stride_grad_input,
            approximate_mode="tanh",
        )
        test_cases.append(test_case)

    # 添加所有测试用例并保存
    test_writer.add_tests(test_cases)
    test_writer.save()


if __name__ == "__main__":
    _TENSOR_DTYPES_ = [np.float32, np.float16, bfloat16]
    dtype_filename_map = {
        np.float32: "gelu_backward_f32.gguf",
        np.float16: "gelu_backward_f16.gguf",
        bfloat16: "gelu_backward_bf16.gguf",
    }

    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)
