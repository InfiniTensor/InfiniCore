from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def add(
    a: np.ndarray,
    b: np.ndarray,
):
    return a + b

def process_tensor(a, b, stride_a=None, stride_b=None):
    def normalize_stride(tensor, stride):
        if stride:
            stride = tuple(reversed(stride))
            if any(s == 0 for s in stride):
                tensor = as_strided(tensor, shape=tensor.shape,
                                    strides=tuple(s * tensor.itemsize for s in stride))
        element_strides = tuple(s // tensor.itemsize for s in tensor.strides)
        slices = tuple(slice(0, 1) if s == 0 else slice(None) for s in element_strides)
        return tensor, tensor[slices]

    a_rearranged, a_unique = normalize_stride(a, stride_a)
    b_rearranged, b_unique = normalize_stride(b, stride_b)

    return a_rearranged, b_rearranged, a_unique, b_unique

class AddTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: np.ndarray,
        a_rearranged:np.ndarray,
        stride_a: List[int] | None,
        shape_a: List[int] | None,
        b: np.ndarray,
        b_rearranged:np.ndarray,
        stride_b: List[int] | None,
        shape_b: List[int] | None,
        c: np.ndarray,
        stride_c: List[int] | None,
        shape_c: List[int] | None,

    ):
        super().__init__("add")
        self.a = a
        self.a_rearranged = a_rearranged
        self.stride_a = stride_a
        self.shape_a = shape_a
        self.b = b
        self.b_rearranged = b_rearranged
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
        ans = add(
            self.a_rearranged.astype(np.float64),
            self.b_rearranged.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("add.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, a_stride, b_stride, c_stride
        ((13, 4), None, None, gguf_strides(20, 1)),
        ((13, 4), gguf_strides(10, 1), gguf_strides(10, 1), gguf_strides(10, 1)),
        ((13, 4), gguf_strides(0, 1), None, gguf_strides(4, 1)),
        ((13, 4, 4), None, None, gguf_strides(20, 4, 1)),
        ((13, 4, 4), gguf_strides(20, 4, 1), gguf_strides(20, 4, 1), gguf_strides(20, 4, 1)),
        ((13, 4, 4), gguf_strides(4, 0, 1), gguf_strides(0, 4, 1), gguf_strides(16, 4, 1)),
        ((16, 5632), None, None, gguf_strides(13312, 1)),
        ((16, 5632), gguf_strides(13312, 1), gguf_strides(13312, 1), gguf_strides(13312, 1)),
        ((4, 4, 5632), None, None, gguf_strides(22528, 5632, 1)),
        ((4, 4, 5632), gguf_strides(45056, 5632, 1), gguf_strides(45056, 5632, 1), gguf_strides(45056, 5632, 1)),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for shape, stride_a, stride_b, stride_c in _TEST_CASES_:
            a = np.random.rand(*shape).astype(dtype)
            b = np.random.rand(*shape).astype(dtype)
            zero_shape = tuple(0 for _ in shape)
            c = np.empty(zero_shape, dtype=dtype)

            a_re, b_re, a_unique, b_unique = process_tensor(a, b, stride_a, stride_b)

            test_case = AddTestCase(
                a=a_unique,
                a_rearranged=a_re,
                stride_a=stride_a,
                shape_a=shape,
                b=b_unique,
                b_rearranged=b_re,
                stride_b=stride_b,
                shape_b=shape,
                c=c,
                stride_c=stride_c,
                shape_c=shape,
            )
            test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()