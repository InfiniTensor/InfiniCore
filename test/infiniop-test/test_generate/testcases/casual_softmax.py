from ast import List
import numpy as np
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def causal_softmax(x):
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    mask = np.tril(np.ones_like(x), k=-1)
    mask = np.flip(mask, axis=(-2, -1))
    masked = np.where(mask == 1, -np.inf, x)
    exp_values = np.exp(masked - np.max(masked, axis=-1, keepdims=True))
    softmax_result = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    return softmax_result


def random_tensor(shape, dtype):
    rate = 1e-3
    var = 0.5 * rate  # 数值范围在[-5e-4, 5e-4]
    return rate * np.random.rand(*shape).astype(dtype) - var


class CausalSoftmaxTestCase(InfiniopTestCase):
    def __init__(
        self,
        data: np.ndarray,
        stride: List[int] | None,
    ):
        super().__init__("causal_softmax")
        self.data = data
        self.stride = stride

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        if self.stride is not None:
            test_writer.add_array(test_writer.gguf_key("data.strides"), self.stride)
        test_writer.add_tensor(
            test_writer.gguf_key("data"),
            self.data,
            raw_dtype=np_dtype_to_ggml(self.data.dtype),
        )
        ans = causal_softmax(
            self.data.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("causal_softmax.gguf")
    # x_shape, x_stride
    test_cases = [
        CausalSoftmaxTestCase(
            random_tensor((3, 3), np.float32),
            None,
        ),
        CausalSoftmaxTestCase(
            random_tensor((32, 512), np.float32),
            gguf_strides(1024, 1),
        ),
        CausalSoftmaxTestCase(
            random_tensor((32, 5, 5), np.float32),
            None,
        ),
        CausalSoftmaxTestCase(
            random_tensor((32, 20, 512), np.float32),
            None,
        ),
        CausalSoftmaxTestCase(
            random_tensor((32, 20, 512), np.float32),
            gguf_strides(20480, 512, 1),
        ),
        CausalSoftmaxTestCase(
            random_tensor((9, 10, 1024), np.float32),
            None,
        ),
        CausalSoftmaxTestCase(
            random_tensor((32, 120), np.float32),
            None,
        ),
        CausalSoftmaxTestCase(
            random_tensor((5, 9, 200), np.float32),
            gguf_strides(1800, 200, 1),
        ),
        CausalSoftmaxTestCase(
            random_tensor((3, 3), np.float16),
            None,
        ),
        CausalSoftmaxTestCase(
            random_tensor((32, 512), np.float16),
            gguf_strides(1024, 1),
        ),
        CausalSoftmaxTestCase(
            random_tensor((32, 5, 5), np.float16),
            None,
        ),
        CausalSoftmaxTestCase(
            random_tensor((32, 20, 512), np.float16),
            None,
        ),
        CausalSoftmaxTestCase(
            random_tensor((32, 20, 512), np.float16),
            gguf_strides(20480, 512, 1),
        ),
        CausalSoftmaxTestCase(
            random_tensor((9, 10, 1024), np.float16),
            None,
        ),
        CausalSoftmaxTestCase(
            random_tensor((32, 120), np.float16),
            None,
        ),
        CausalSoftmaxTestCase(
            random_tensor((5, 9, 200), np.float16),
            gguf_strides(1800, 200, 1),
        ),
    ]
    test_writer.add_tests(test_cases)
    test_writer.save()
