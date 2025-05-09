import numpy as np
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def sigmoid(
        x: np.ndarray,
):
    return 1 / (1 + np.exp(-x))


def random_tensor(shape, dtype):
    rate = 1e-3
    var = 0.5 * rate  # 数值范围在[-5e-4, 5e-4]
    return rate * np.random.rand(*shape).astype(dtype) - var


class SigmoidTestCase(InfiniopTestCase):
    def __init__(
            self,
            x: np.ndarray,
            stride_x: List[int] | None,
            y: np.ndarray,
            stride_y: List[int] | None,
    ):
        super().__init__("sigmoid")
        self.x = x
        self.stride_x = stride_x
        self.y = y
        self.stride_y = stride_y

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        if self.stride_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), self.stride_x)
        if self.stride_y is not None:
            test_writer.add_array(test_writer.gguf_key("y.strides"), self.stride_y)

        test_writer.add_tensor(
            test_writer.gguf_key("x"), self.x, raw_dtype=np_dtype_to_ggml(self.x.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("y"), self.y, raw_dtype=np_dtype_to_ggml(self.y.dtype)
        )

        x_fp64 = self.x.astype(np.float64)
        ans_fp64 = 1 / (1 + np.exp(x_fp64))
        ans = sigmoid(self.x)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=np_dtype_to_ggml(ans.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_fp64"),
            ans_fp64,
            raw_dtype=np_dtype_to_ggml(ans_fp64.dtype),
        )


if __name__ == '__main__':
    test_writer = InfiniopTestWriter("sigmoid.gguf")
    test_cases = [
        SigmoidTestCase(
            random_tensor((2, 3), np.float32),
            gguf_strides(3, 1),
            random_tensor((2, 3), np.float32),
            gguf_strides(3, 1),
        ),
        SigmoidTestCase(
            random_tensor((2, 3), np.float16),
            gguf_strides(1, 2),
            random_tensor((2, 3), np.float16),
            gguf_strides(1, 2),
        ),
        SigmoidTestCase(
            random_tensor((2, 3), np.float64),
            gguf_strides(3, 1),
            random_tensor((2, 3), np.float64),
            gguf_strides(1, 2),
        ),
        SigmoidTestCase(
            random_tensor((4, 6), np.float16),
            gguf_strides(1, 4),
            random_tensor((4, 6), np.float16),
            gguf_strides(6, 1),
        ),
        SigmoidTestCase(
            random_tensor((1, 2048), np.float16),
            gguf_strides(1, 1),
            random_tensor((1, 2048), np.float16),
            gguf_strides(1, 1),
        ),
        SigmoidTestCase(
            random_tensor((2048, 2048), np.float32),
            None,
            random_tensor((2048, 2048), np.float32),
            None,
        ),
        SigmoidTestCase(
            random_tensor((2, 4, 2048), np.float16),
            gguf_strides(4 * 2048, 2048, 1),
            random_tensor((2, 4, 2048), np.float16),
            gguf_strides(4 * 2048, 2048, 1),
        ),
        SigmoidTestCase(
            random_tensor((2, 4, 2048), np.float32),
            gguf_strides(1, 2, 2 * 4),
            random_tensor((2, 4, 2048), np.float32),
            gguf_strides(1, 2, 2 * 4),
        ),
        SigmoidTestCase(
            random_tensor((2048, 2560), np.float32),
            gguf_strides(2560, 1),
            random_tensor((2048, 2560), np.float32),
            gguf_strides(2560, 1),
        ),
        SigmoidTestCase(
            random_tensor((4, 48, 64), np.float16),
            gguf_strides(64 * 48, 64, 1),
            random_tensor((4, 48, 64), np.float16),
            None
        ),
        SigmoidTestCase(
            random_tensor((4, 48, 64), np.float32),
            None,
            random_tensor((4, 48, 64), np.float32),
            gguf_strides(48 * 64, 64, 1),
        )
    ]
    test_writer.add_tests(test_cases)
    test_writer.save()
