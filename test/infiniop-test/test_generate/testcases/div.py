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


def div(
    a: np.ndarray, b: np.ndarray, rounding_mode: Optional[str] = None
) -> np.ndarray:
    """
    执行除法操作，支持不同的取整模式
    Args:
        a: 被除数张量
        b: 除数张量
        rounding_mode: 取整模式 (None, "trunc" 或 "floor")
    Returns:
        除法结果张量
    """
    result = a.astype(np.float64) / b.astype(np.float64)

    if rounding_mode == "trunc":
        # 向零取整
        result = np.trunc(result)
    elif rounding_mode == "floor":
        # 向下取整
        result = np.floor(result)

    return result


def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """
    生成指定形状和数据类型的随机张量
    """
    return np.random.randn(*shape).astype(dtype)


class DivTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: np.ndarray,
        shape_a: List[int] | None,
        stride_a: List[int] | None,
        b: np.ndarray,
        shape_b: List[int] | None,
        stride_b: List[int] | None,
        c: np.ndarray,
        shape_c: List[int] | None,
        stride_c: List[int] | None,
        rounding_mode: Optional[str] = None,
    ):
        super().__init__("div")
        self.a = a
        self.shape_a = shape_a
        self.stride_a = stride_a
        self.b = b
        self.shape_b = shape_b
        self.stride_b = stride_b
        self.c = c
        self.shape_c = shape_c
        self.stride_c = stride_c
        self.rounding_mode = rounding_mode

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # 添加形状信息
        if self.shape_a is not None:
            test_writer.add_array(test_writer.gguf_key("a.shape"), self.shape_a)
        if self.shape_b is not None:
            test_writer.add_array(test_writer.gguf_key("b.shape"), self.shape_b)
        if self.shape_c is not None:
            test_writer.add_array(test_writer.gguf_key("c.shape"), self.shape_c)

        # 添加步长信息
        if self.stride_a is not None:
            test_writer.add_array(
                test_writer.gguf_key("a.strides"), gguf_strides(*self.stride_a)
            )
        if self.stride_b is not None:
            test_writer.add_array(
                test_writer.gguf_key("b.strides"), gguf_strides(*self.stride_b)
            )
        test_writer.add_array(
            test_writer.gguf_key("c.strides"),
            gguf_strides(
                *(
                    self.stride_c
                    if self.stride_c is not None
                    else contiguous_gguf_strides(self.shape_c)
                )
            ),
        )

        # 添加张量数据
        test_writer.add_tensor(
            test_writer.gguf_key("a"), self.a, raw_dtype=np_dtype_to_ggml(self.a.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("b"), self.b, raw_dtype=np_dtype_to_ggml(self.b.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("c"), self.c, raw_dtype=np_dtype_to_ggml(self.c.dtype)
        )

        # 计算并添加预期结果
        ans = div(self.a, self.b, self.rounding_mode)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


def gen_gguf(dtype: np.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    # 测试用例配置
    _TEST_CASES_ = [
        # (shape, stride_a, stride_b, stride_c)
        ((10,), None, None, None),
        ((5, 10), None, None, None),
        ((3, 4, 5), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4), (0, 1), None, None),
        ((16, 16), None, None, None),
        ((1, 100), None, None, None),
        ((100, 1), None, None, None),
        ((2, 3, 4, 5), None, None, None),
        ((16, 512), None, None, None),
        ((4, 4, 512), None, None, None),
    ]

    # 生成测试用例
    for shape, stride_a, stride_b, stride_c in _TEST_CASES_:
        # 生成随机张量
        a = random_tensor(shape, dtype)
        b = random_tensor(shape, dtype)

        # 确保除数不包含零（避免除以零）
        b = np.where(np.abs(b) < 1e-6, 1e-6 * np.sign(b), b).astype(dtype)

        # 处理零步长情况
        a = process_zero_stride_tensor(a, stride_a)
        b = process_zero_stride_tensor(b, stride_b)

        # 创建输出张量（初始为空）
        c = np.empty(tuple(0 for _ in shape), dtype=dtype)

        # 创建测试用例
        test_case = DivTestCase(
            a=a,
            shape_a=shape,
            stride_a=stride_a,
            b=b,
            shape_b=shape,
            stride_b=stride_b,
            c=c,
            shape_c=shape,
            stride_c=stride_c,
        )
        test_cases.append(test_case)

    # 添加所有测试用例并保存
    test_writer.add_tests(test_cases)
    test_writer.save()


if __name__ == "__main__":
    _TENSOR_DTYPES_ = [np.float32, np.float16, bfloat16]
    dtype_filename_map = {
        np.float32: "div_f32.gguf",
        np.float16: "div_f16.gguf",
        bfloat16: "div_bf16.gguf",
    }

    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)
