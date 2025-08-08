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


def equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """模拟 torch.eq 行为：形状和数据类型必须完全一致，元素值逐元素相等"""
    return np.equal(a, b)


def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """
    生成指定形状和数据类型的随机张量
    """
    return np.random.randn(*shape).astype(dtype)


class EqualTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: np.ndarray,
        shape_a: List[int],
        stride_a: List[int] | None,
        b: np.ndarray,
        shape_b: List[int],
        stride_b: List[int] | None,
        c: np.ndarray,
        shape_c: List[int],
        stride_c: List[int] | None,
    ):
        super().__init__("equal")
        self.a = a
        self.shape_a = shape_a
        self.stride_a = stride_a
        self.b = b
        self.shape_b = shape_b
        self.stride_b = stride_b
        self.c = c
        self.shape_c = shape_c
        self.stride_c = stride_c

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
        ans = equal(self.a, self.b)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans.astype(np.bool),
            raw_dtype=gguf.GGMLQuantizationType.Q8_K,
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("equal.gguf")
    test_cases = []

    # 测试用例配置
    _TEST_CASES_ = [
        ((10,), None, None, None),
        ((5, 10), None, None, None),
        ((3, 4, 5), None, None, None),
        ((16, 16), None, None, None),
        ((1, 100), None, None, None),
        ((100, 1), None, None, None),
        ((2, 3, 4, 5), None, None, None),
        ((13, 4), (10, 1), (10, 1), None),
        ((13, 4), (0, 1), (1, 0), None),
        ((5, 1), (1, 10), None, None),
        ((3, 1, 5), (0, 5, 1), None, None),
        ((10, 1), (5, 10), None, None),
        ((10, 5), (100, 1), None, None),
    ]

    _TENSOR_DTYPES_ = [
        # 浮点类型
        np.float64,
        np.float32,
        np.float16,
        bfloat16,
        # 整数类型
        np.int64,
        np.int32,
        np.int16,
        np.int8,
    ]
    for dtype in _TENSOR_DTYPES_:
        for shape, stride_a, stride_b, stride_c in _TEST_CASES_:
            # 生成随机张量
            a = random_tensor(shape, dtype)
            b = random_tensor(shape, dtype)

            # 处理零步长情况
            a = process_zero_stride_tensor(a, stride_a)
            b = process_zero_stride_tensor(b, stride_b)

            # 创建输出张量（初始为空）
            c = np.empty(tuple(0 for _ in shape), dtype=np.bool)

            # 创建测试用例
            test_case = EqualTestCase(
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

    # 保存所有测试用例
    test_writer.add_tests(test_cases)
    test_writer.save()
