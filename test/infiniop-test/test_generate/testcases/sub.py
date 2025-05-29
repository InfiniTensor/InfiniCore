from ast import List
import numpy as np
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def sub(
    a: np.ndarray,
    b: np.ndarray,
):
    return a - b


def random_tensor(shape, dtype):
    rate = 1e-3
    var = 0.5 * rate  # Numerical range in [-5e-4, 5e-4]
    return rate * np.random.rand(*shape).astype(dtype) - var


class SubTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: np.ndarray,
        stride_a: List[int] | None,
        b: np.ndarray,
        stride_b: List[int] | None,
        c: np.ndarray,
        stride_c: List[int] | None,
    ):
        super().__init__("sub")
        self.a = a
        self.stride_a = stride_a
        self.b = b
        self.stride_b = stride_b
        self.c = c
        self.stride_c = stride_c

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        if self.stride_a is not None:
            test_writer.add_array(test_writer.gguf_key("a.strides"), self.stride_a)
        if self.stride_b is not None:
            test_writer.add_array(test_writer.gguf_key("b.strides"), self.stride_b)
        if self.stride_c is not None:
            test_writer.add_array(test_writer.gguf_key("c.strides"), self.stride_c)
        test_writer.add_tensor(
            test_writer.gguf_key("a"), self.a, raw_dtype=np_dtype_to_ggml(self.a.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("b"), self.b, raw_dtype=np_dtype_to_ggml(self.b.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("c"), self.c, raw_dtype=np_dtype_to_ggml(self.c.dtype)
        )
        ans = sub(
            self.a.astype(np.float64),
            self.b.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("sub.gguf")
    # a, stride_a, b, stride_b, c, stride_c
    test_cases = [
        SubTestCase(
            random_tensor((4, 5), np.float32),
            None,
            random_tensor((4, 5), np.float32),
            None,
            random_tensor((4, 5), np.float32),
            None,
        ),
        SubTestCase(
            random_tensor((4, 5), np.float32),
            gguf_strides(1, 4),
            random_tensor((4, 5), np.float32),
            gguf_strides(1, 4),
            random_tensor((4, 5), np.float32),
            gguf_strides(1, 4),
        ),
        SubTestCase(
            random_tensor((4, 5), np.float16),
            None,
            random_tensor((4, 5), np.float16),
            None,
            random_tensor((4, 5), np.float16),
            None,
        ),
        SubTestCase(
            random_tensor((4, 5), np.float16),
            gguf_strides(1, 4),
            random_tensor((4, 5), np.float16),
            gguf_strides(1, 4),
            random_tensor((4, 5), np.float16),
            gguf_strides(1, 4),
        ),
        SubTestCase(
            random_tensor((1, 2048), np.float16),
            gguf_strides(1, 2048),
            random_tensor((1, 2048), np.float16),
            gguf_strides(1, 2048),
            random_tensor((1, 2048), np.float16),
            gguf_strides(1, 2048),
        ),
        SubTestCase(
            random_tensor((1, 2048), np.float32),
            None,
            random_tensor((1, 2048), np.float32),
            None,
            random_tensor((1, 2048), np.float32),
            None,
        ),
        SubTestCase(
            random_tensor((2, 4, 2048), np.float16),
            None,
            random_tensor((2, 4, 2048), np.float16),
            None,
            random_tensor((2, 4, 2048), np.float16),
            None,
        ),
        SubTestCase(
            random_tensor((2, 4, 2048), np.float32),
            None,
            random_tensor((2, 4, 2048), np.float32),
            None,
            random_tensor((2, 4, 2048), np.float32),
            None,
        ),
        SubTestCase(
            random_tensor((6, 2048), np.float32),
            gguf_strides(1, 2048),
            random_tensor((6, 2048), np.float32),
            gguf_strides(1, 2048),
            random_tensor((6, 2048), np.float32),
            gguf_strides(1, 2048),
        ),
        SubTestCase(
            random_tensor((4, 48, 64), np.float16),
            None,
            random_tensor((4, 48, 64), np.float16),
            None,
            random_tensor((4, 48, 64), np.float16),
            None,
        ),
        SubTestCase(
            random_tensor((4, 48, 64), np.float32),
            None,
            random_tensor((4, 48, 64), np.float32),
            None,
            random_tensor((4, 48, 64), np.float32),
            None,
        ),
    ]
    test_writer.add_tests(test_cases)
    test_writer.save()
