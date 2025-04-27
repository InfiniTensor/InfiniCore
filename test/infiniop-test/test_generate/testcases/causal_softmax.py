from ast import List
import numpy as np
import gguf
from typing import List
from enum import Enum, auto

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
        x: np.ndarray,
        y: np.ndarray,
        stride_x: List[int] | None,
        stride_y: List[int] | None,
    ):
        super().__init__("causal_softmax")
        self.x = x
        self.y = y
        self.stride_x = stride_x
        self.stride_y = stride_y

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        if self.stride_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), self.stride_x)
        if self.stride_y is not None:
            test_writer.add_array(test_writer.gguf_key("y.strides"), self.stride_y)
        test_writer.add_tensor(
            test_writer.gguf_key("x"),
            self.x,
            raw_dtype=np_dtype_to_ggml(self.x.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("y"),
            self.y,
            raw_dtype=np_dtype_to_ggml(self.y.dtype),
        )
        ans = causal_softmax(
            self.x.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


# ==============================================================================
#  Configuration
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, x_stride, y_stride
    ((3, 3), None, None),
    ((32, 512), None, None),
    ((32, 512), gguf_strides(1024, 1), gguf_strides(1024, 1)),
    ((32, 5, 5), None, None),
    ((32, 20, 512), None, None),
    ((32, 20, 512), gguf_strides(20480, 512, 1), None),
]

_TENSOR_DTYPES_ = [np.float16, np.float32]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_X,
]


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("causal_softmax.gguf")
    # shape, x_stride, y_stride
    test_cases = []

    for dtype in _TENSOR_DTYPES_:
        for shape, stride_x, stride_y in _TEST_CASES_:
            for inplace in _INPLACE:
                x = random_tensor(shape, dtype)
                if inplace == Inplace.INPLACE_X:
                    y = x
                else:
                    y = random_tensor(shape, dtype)

                test_case = CausalSoftmaxTestCase(
                    x,
                    y,
                    stride_x,
                    stride_y,
                )
                test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()


