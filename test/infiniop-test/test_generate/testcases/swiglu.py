import numpy as np
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def swiglu(
    a: np.ndarray,
    b: np.ndarray,
):
    c = a * b / (1.0 + np.exp(-b))

    return c


class SwiGLUTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: np.ndarray,
        stride_a: List[int] | None,
        shape_a: List[int] | None,
        b: np.ndarray,
        stride_b: List[int] | None,
        shape_b: List[int] | None,
        c: np.ndarray,
        stride_c: List[int] | None,
        shape_c: List[int] | None,

    ):
        super().__init__("swiglu")
        self.a = a
        self.stride_a = stride_a
        self.shape_a = shape_a
        self.b = b
        self.stride_b = stride_b
        self.shape_b = shape_b
        self.c = c
        self.stride_c = stride_c
        self.shape_c = shape_c


    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        if self.stride_a is not None:
            test_writer.add_array(test_writer.gguf_key("a.strides"), self.stride_a)
        if self.stride_b is not None:
            test_writer.add_array(test_writer.gguf_key("b.strides"), self.stride_b)
        if self.stride_c is not None:
            test_writer.add_array(test_writer.gguf_key("c.strides"), self.stride_c)
        if self.shape_a is not None:
            test_writer.add_array(test_writer.gguf_key("a.shape"), self.shape_a)
        if self.shape_b is not None:
            test_writer.add_array(test_writer.gguf_key("b.shape"), self.shape_b)
        if self.shape_c is not None:
            test_writer.add_array(test_writer.gguf_key("c.shape"), self.shape_c)        
        test_writer.add_tensor(
            test_writer.gguf_key("a"), self.a, raw_dtype=np_dtype_to_ggml(self.a.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("b"), self.b, raw_dtype=np_dtype_to_ggml(self.b.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("c"), self.c, raw_dtype=np_dtype_to_ggml(self.c.dtype)
        )
        ans = swiglu(
            self.a.astype(np.float64),
            self.b.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("swiglu.gguf")
    test_cases = [
        SwiGLUTestCase(
            np.random.rand(64, 128).astype(np.float32),
            None,
            (64, 128),
            np.random.rand(64, 128).astype(np.float32),
            None,
            (64, 128),
            np.empty(tuple(0 for _ in (64, 128)), dtype=np.float32),
            gguf_strides(128, 1),
            (64, 128)
        ),
        SwiGLUTestCase(
            np.random.rand(64, 121).astype(np.float32),
            None,
            (64, 121),
            np.random.rand(64, 121).astype(np.float32),
            None,
            (64, 121),
            np.empty(tuple(0 for _ in (64, 121)), dtype=np.float32),
            gguf_strides(121, 1),
            (64, 121),
        ),
        SwiGLUTestCase(
            np.random.rand(15, 512).astype(np.float32),
            None,
            (15, 512),
            np.random.rand(15, 512).astype(np.float32),
            None,
            (15, 512),
            np.empty(tuple(0 for _ in (15, 512)), dtype=np.float32),
            gguf_strides(512, 1),
            (15, 512),
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4).astype(np.float32),
            None,
            (13, 4),
            np.random.rand(13, 4).astype(np.float32),
            None,
            (13, 4),
            np.empty(tuple(0 for _ in (13, 4)), dtype=np.float32),
            gguf_strides(4, 1),
            (13, 4)
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4).astype(np.float16),
            None,
            (13, 4),
            np.random.rand(13, 4).astype(np.float16),
            None,
            (13, 4),
            np.empty(tuple(0 for _ in (13, 4)), dtype=np.float16),
            gguf_strides(4, 1),
            (13, 4),
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4).astype(np.float32),
            gguf_strides(10, 1),
            (13, 4),
            np.random.rand(13, 4).astype(np.float32),
            gguf_strides(10, 1),
            (13, 4),
            np.empty(tuple(0 for _ in (13, 4)), dtype=np.float32),
            gguf_strides(10, 1),
            (13, 4),
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4).astype(np.float16),
            gguf_strides(10, 1),
            (13, 4),
            np.random.rand(13, 4).astype(np.float16),
            gguf_strides(10, 1),
            (13, 4),
            np.empty(tuple(0 for _ in (13, 4)), dtype=np.float16),
            gguf_strides(10, 1),
            (13, 4),
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4, 4).astype(np.float32),
            None,
            (13, 4, 4),
            np.random.rand(13, 4, 4).astype(np.float32),
            None,
            (13, 4, 4),
            np.empty(tuple(0 for _ in (13, 4, 4)), dtype=np.float32),
            gguf_strides(16, 4, 1),
            (13, 4, 4),
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4, 4).astype(np.float16),
            None,
            (13, 4, 4),
            np.random.rand(13, 4, 4).astype(np.float16),
            None,
            (13, 4, 4),
            np.empty(tuple(0 for _ in (13, 4, 4)), dtype=np.float16),
            gguf_strides(16, 4, 1),
            (13, 4, 4),
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4, 4).astype(np.float32),
            gguf_strides(20, 4, 1),
            (13, 4, 4),
            np.random.rand(13, 4, 4).astype(np.float32),
            gguf_strides(20, 4, 1),
            (13, 4, 4),
            np.empty(tuple(0 for _ in (13, 4, 4)), dtype=np.float32),
            gguf_strides(20, 4, 1),
            (13, 4, 4),
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4, 4).astype(np.float16),
            gguf_strides(20, 4, 1),
            (13, 4, 4),
            np.random.rand(13, 4, 4).astype(np.float16),
            gguf_strides(20, 4, 1),
            (13, 4, 4),
            np.empty(tuple(0 for _ in (13, 4, 4)), dtype=np.float16),
            gguf_strides(20, 4, 1),
            (13, 4, 4),
        ),
        SwiGLUTestCase(
            np.random.rand(16, 5632).astype(np.float32),
            None,
            (16, 5632),
            np.random.rand(16, 5632).astype(np.float32),
            None,
            (16, 5632),
            np.empty(tuple(0 for _ in (16, 5632)), dtype=np.float32),
            gguf_strides(5632, 1),
            (16, 5632),
        ),
        SwiGLUTestCase(
            np.random.rand(16, 5632).astype(np.float16),
            None,
            (16, 5632),
            np.random.rand(16, 5632).astype(np.float16),
            None,
            (16, 5632),
            np.empty(tuple(0 for _ in (16, 5632)), dtype=np.float16),
            gguf_strides(5632, 1),
            (16, 5632),
        ),
        SwiGLUTestCase(
            np.random.rand(16, 5632).astype(np.float32),
            gguf_strides(13312, 1),
            (16, 5632),
            np.random.rand(16, 5632).astype(np.float32),
            gguf_strides(13312, 1),
            (16, 5632),
            np.empty(tuple(0 for _ in (16, 5632)), dtype=np.float32),
            gguf_strides(13312, 1),
            (16, 5632),
        ),
        SwiGLUTestCase(
            np.random.rand(16, 5632).astype(np.float16),
            gguf_strides(13312, 1),
            (16, 5632),
            np.random.rand(16, 5632).astype(np.float16),
            gguf_strides(13312, 1),
            (16, 5632),
            np.empty(tuple(0 for _ in (16, 5632)), dtype=np.float16),
            gguf_strides(13312, 1),
            (16, 5632),
        ),
        SwiGLUTestCase(
            np.random.rand(16, 5632).astype(np.float32),
            gguf_strides(5632, 1),
            (16, 5632),
            np.random.rand(16, 5632).astype(np.float32),
            gguf_strides(5632, 1),
            (16, 5632),
            np.empty(tuple(0 for _ in (16, 5632)), dtype=np.float32),
            gguf_strides(1, 16),
            (16, 5632),
        ),
        SwiGLUTestCase(
            np.random.rand(16, 5632).astype(np.float16),
            gguf_strides(5632, 1),
            (16, 5632),
            np.random.rand(16, 5632).astype(np.float16),
            gguf_strides(5632, 1),
            (16, 5632),
            np.empty(tuple(0 for _ in (16, 5632)), dtype=np.float16),
            gguf_strides(1, 16),
            (16, 5632),
        ),
        SwiGLUTestCase(
            np.random.rand(2, 3, 400).astype(np.float32),
            gguf_strides(1200, 400, 1),
            (2, 3, 400),
            np.random.rand(2, 3, 400).astype(np.float32),
            gguf_strides(1200, 400, 1),
            (2, 3, 400),
            np.empty(tuple(0 for _ in (2, 3, 400)), dtype=np.float32),
            gguf_strides(1, 2, 6),
            (2, 3, 400),
        ),
        SwiGLUTestCase(
            np.random.rand(2, 3, 400).astype(np.float16),
            gguf_strides(1200, 400, 1),
            (2, 3, 400),
            np.random.rand(2, 3, 400).astype(np.float16),
            gguf_strides(1200, 400, 1),
            (2, 3, 400),
            np.empty(tuple(0 for _ in (2, 3, 400)), dtype=np.float16),
            gguf_strides(1, 2, 6),
            (2, 3, 400),
        ),
        SwiGLUTestCase(
            np.random.rand(4, 4, 5632).astype(np.float32),
            None,
            (4, 4, 5632),
            np.random.rand(4, 4, 5632).astype(np.float32),
            None,
            (4, 4, 5632),
            np.empty(tuple(0 for _ in (4, 4, 5632)), dtype=np.float32),
            gguf_strides(22528, 5632, 1),
            (4, 4, 5632),
        ),
        SwiGLUTestCase(
            np.random.rand(4, 4, 5632).astype(np.float16),
            None,
            (4, 4, 5632),
            np.random.rand(4, 4, 5632).astype(np.float16),
            None,
            (4, 4, 5632),
            np.empty(tuple(0 for _ in (4, 4, 5632)), dtype=np.float16),
            gguf_strides(22528, 5632, 1),
            (4, 4, 5632),
        ),
        SwiGLUTestCase(
            np.random.rand(4, 4, 5632).astype(np.float32),
            gguf_strides(45056, 5632, 1),
            (4, 4, 5632),
            np.random.rand(4, 4, 5632).astype(np.float32),
            gguf_strides(45056, 5632, 1),
            (4, 4, 5632),
            np.empty(tuple(0 for _ in (4, 4, 5632)), dtype=np.float32),
            gguf_strides(45056, 5632, 1),
            (4, 4, 5632),
        ),
        SwiGLUTestCase(
            np.random.rand(4, 4, 5632).astype(np.float16),
            gguf_strides(45056, 5632, 1),
            (4, 4, 5632),
            np.random.rand(4, 4, 5632).astype(np.float16),
            gguf_strides(45056, 5632, 1),
            (4, 4, 5632),
            np.empty(tuple(0 for _ in (4, 4, 5632)), dtype=np.float16),
            gguf_strides(45056, 5632, 1),
            (4, 4, 5632),
        ),
    ]
    test_writer.add_tests(test_cases)
    test_writer.save()
