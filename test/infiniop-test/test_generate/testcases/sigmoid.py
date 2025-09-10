import numpy as np
from numpy.lib.stride_tricks import as_strided
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def sigmoid(
        x: np.ndarray,
):
    return 1 / (1 + np.exp(-x))


def random_tensor(shape, dtype):
    rate = 1e-3
    var = 0.5 * rate
    return rate * np.random.rand(*shape).astype(dtype) - var


def process_tensors(a, b, stride_a=None, stride_b=None):
    def normalize_stride(tensor, stride):
        if stride:
            slices = tuple(slice(0, 1) if s == 0 else slice(None) for s in stride)
            return tensor[slices]
        else:
            return tensor

    a_unique = normalize_stride(a, stride_a)
    b_unique = normalize_stride(b, stride_b)
    return a_unique, b_unique


def process_tensor(a, stride_a=None):
    def normalize_stride(tensor, stride):
        if stride:
            slices = tuple(slice(0, 1) if s == 0 else slice(None) for s in stride)
            return tensor[slices]
        else:
            return tensor

    a_unique = normalize_stride(a, stride_a)
    return a_unique


class SigmoidTestCase(InfiniopTestCase):
    def __init__(
            self,
            x: np.ndarray,
            shape_x: List[int] | None,
            stride_x: List[int] | None,
            y: np.ndarray,
            shape_y: List[int] | None,
            stride_y: List[int] | None,
    ):
        super().__init__("sigmoid")
        self.x = x
        self.shape_x = shape_x
        self.stride_x = stride_x

        self.y = y
        self.shape_y = shape_y
        self.stride_y = stride_y

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        if self.shape_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.shape"), self.shape_x)
        if self.shape_y is not None:
            test_writer.add_array(test_writer.gguf_key("y.shape"), self.shape_y)

        if self.stride_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*self.stride_x))

        test_writer.add_array(
            test_writer.gguf_key("y.strides"),
            gguf_strides(*self.stride_y if self.stride_y is not None else contiguous_gguf_strides(self.shape_y))
        )

        test_writer.add_tensor(
            test_writer.gguf_key("x"), self.x, raw_dtype=np_dtype_to_ggml(self.x.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("y"), self.y, raw_dtype=np_dtype_to_ggml(self.y.dtype)
        )

        input_x = self.x.astype(np.float64)
        if (self.stride_x is not None) and (0 in self.stride_x):
            typesize = np.dtype(input_x.dtype).itemsize
            new_strides_bytes = tuple(x * typesize for x in self.stride_x)
            input_x = as_strided(x=input_x, shape=self.shape_x, strides=new_strides_bytes)

        ans = sigmoid(input_x)

        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == '__main__':
    test_writer = InfiniopTestWriter("sigmoid.gguf")

    test_cases = []
    _TEST_CASES_ = [
        # shape, x_stride, y_stride
        ((13, 4), None, None),
        ((13, 4), (10, 1), (10, 1)),
        ((13, 4), (0, 1), None),
        ((13, 4, 4), None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
        ((13, 4, 4), (4, 0, 1), None),
        ((16, 5632), None, None),
        ((16, 5632), (13312, 1), (13312, 1)),
        ((4, 4, 5632), None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
    ]
    _TENSOR_DTYPES_ = [np.float16, np.float32]

    for dtype in _TENSOR_DTYPES_:
        for shape, stride_x, stride_y in _TEST_CASES_:
            x = np.random.rand(*shape).astype(dtype)
            y = np.empty(tuple(0 for _ in shape), dtype=dtype)

            x = process_zero_stride_tensor(x, stride_x)
            test_case = SigmoidTestCase(x=x,
                                        shape_x=shape,
                                        stride_x=stride_x,
                                        y=y,
                                        shape_y=shape,
                                        stride_y=stride_y)

            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
