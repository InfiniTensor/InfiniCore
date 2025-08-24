from ast import List
import numpy as np
import gguf
from typing import List
from enum import Enum, auto

from .. import (
    InfiniopTestWriter,
    InfiniopTestCase,
    np_dtype_to_ggml,
    gguf_strides,
    contiguous_gguf_strides,
)


def reduce_mean(x, dim):
    if isinstance(x, np.float64):
        return x
    return x.mean(axis=dim, keepdims=True)


def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, shape).astype(dtype) * 0.001


class ReduceMeanTestCase(InfiniopTestCase):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        shape_x: List[int] | None,
        shape_y: List[int] | None,
        stride_x: List[int] | None,
        stride_y: List[int] | None,
        dim: int = 0,
    ):
        super().__init__("reduce_mean")
        self.x = x
        self.y = y
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.dim = dim

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        print(self.shape_y, self.shape_x, self.stride_y, self.stride_x, self.dim)
        if self.shape_x is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_x)
        if self.shape_y is not None:
            test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape_y)
        if self.stride_x is not None:
            test_writer.add_array(
                test_writer.gguf_key("input.strides"), gguf_strides(*self.stride_x)
            )
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(
                *(
                    self.stride_y
                    if self.stride_y is not None
                    else contiguous_gguf_strides(self.shape_y)
                )
            ),
        )
        test_writer.add_uint64(test_writer.gguf_key("dim"), self.dim)
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            self.x,
            raw_dtype=np_dtype_to_ggml(self.x.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            self.y,
            raw_dtype=np_dtype_to_ggml(self.y.dtype),
        )
        ans = reduce_mean(self.x.astype(np.float64), self.dim)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("reduce_mean.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # y_shape, x_shape, y_stride, x_stride, dim
        # ((0,), (0,), (0,), (0,), 0),
        ((1,), (32,), None, None, 0),
        ((1, 4), (1, 4), None, None, 0),
        ((1, 1), (1, 4), None, None, 1),
        ((16, 1), (16, 2048), None, None, 1),
        ((1, 16), (2048, 16), None, None, 0),
        ((16, 1), (16, 2048), (4096, 1), (4096, 1), 1),
        ((1, 2048), (16, 2048), (4096, 1), (4096, 1), 0),
        ((4, 4, 1), (4, 4, 2048), None, None, 2),
        ((1, 4, 4), (2048, 4, 4), None, None, 0),
        ((4, 1, 4), (4, 2048, 4), (45056, 5632, 1), (32768, 8, 1), 1),
        ((1, 8, 4, 8), (16, 8, 4, 8), (256, 32, 8, 1), (256, 32, 8, 1), 0),
    ]
    _TENSOR_DTYPES_ = [np.float16, np.float32]

    for dtype in _TENSOR_DTYPES_:
        for shape_y, shape_x, stride_y, stride_x, dim in _TEST_CASES_:
            x = random_tensor(shape_x, dtype)
            y = np.empty(tuple(0 for _ in shape_y), dtype=dtype)
            test_case = ReduceMeanTestCase(
                x,
                y,
                shape_x,
                shape_y,
                stride_x,
                stride_y,
                dim,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
