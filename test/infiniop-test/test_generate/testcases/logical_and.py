from typing import List

import gguf
import numpy as np

from .. import (
    InfiniopTestCase,
    InfiniopTestWriter,
    contiguous_gguf_strides,
    gguf_strides,
    np_dtype_to_ggml,
    process_zero_stride_tensor,
)


def logical_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """执行元素级逻辑与操作，非零值为True，零值为False"""
    return np.logical_and(a.astype(bool), b.astype(bool))


def random_logical_tensor(shape: tuple):
    """生成包含随机布尔值（0/1）的张量"""
    # 布尔类型：直接生成True/False
    return np.random.choice([True, False], size=shape)


class LogicalAndTestCase(InfiniopTestCase):
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
        super().__init__("logical_and")
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
        ans = logical_and(self.a, self.b)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans.astype(np.bool),
            raw_dtype=gguf.GGMLQuantizationType.Q8_K,
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("logical_and.gguf")
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
        ((10, 5), (10, 1), None, None),
    ]

    for shape, stride_a, stride_b, stride_c in _TEST_CASES_:
        # 生成随机张量
        a = random_logical_tensor(shape)
        b = random_logical_tensor(shape)

        # 处理零步长情况
        a = process_zero_stride_tensor(a, stride_a)
        b = process_zero_stride_tensor(b, stride_b)

        # 创建输出张量（初始为空）
        c = np.empty(tuple(0 for _ in shape), dtype=np.bool)

        # 创建测试用例
        test_case = LogicalAndTestCase(
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
