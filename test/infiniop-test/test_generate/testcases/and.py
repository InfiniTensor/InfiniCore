from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor
import torch

def op_and(
    a: np.ndarray,
    b: np.ndarray,
):
    return torch.logical_and(
        torch.from_numpy(a),
        torch.from_numpy(b)
    ).detach().numpy()


class AndTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: np.ndarray,
        shape_a: List[int] | None,
        stride_a: List[int] | None,
        b: np.ndarray,
        shape_b: List[int] | None,
        stride_b: List[int] | None,
        c: np.ndarray,
        shape_c: List[int] | None,
        stride_c: List[int] | None,

    ):
        super().__init__("op_and")
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
        test_writer.add_array(test_writer.gguf_key("a.shape"), self.shape_a)
        test_writer.add_array(test_writer.gguf_key("b.shape"), self.shape_b)
        test_writer.add_array(test_writer.gguf_key("c.shape"), self.shape_c)
        if self.stride_a is not None:
            test_writer.add_array(test_writer.gguf_key("a.strides"), gguf_strides(*self.stride_a))
        if self.stride_b is not None:
            test_writer.add_array(test_writer.gguf_key("b.strides"), gguf_strides(*self.stride_b))
        test_writer.add_array(
            test_writer.gguf_key("c.strides"),
            gguf_strides(*self.stride_c if self.stride_c is not None else contiguous_gguf_strides(self.shape_c))
        )
        test_writer.add_tensor(
            test_writer.gguf_key("a"), self.a, raw_dtype=np_dtype_to_ggml(self.a.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("b"), self.b, raw_dtype=np_dtype_to_ggml(self.b.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("c"), self.c, raw_dtype=np_dtype_to_ggml(self.c.dtype)
        )
        ans = op_and(
            self.a,
            self.b,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.Q8_K
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("op_and.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, a_stride, b_stride, c_stride
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4, 4), None, None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
        ((16, 5632), None, None, None),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
        ((4, 4, 5632), None, None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
    ]
    for shape, stride_a, stride_b, stride_c in _TEST_CASES_:
        a = np.random.randint(low=0, high=2, size=shape).astype(np.bool_)
        b = np.random.randint(low=0, high=2, size=shape).astype(np.bool_)
        c = np.empty(tuple(0 for _ in shape), dtype=np.bool_)
        a = process_zero_stride_tensor(a, stride_a)
        b = process_zero_stride_tensor(b, stride_b)
        test_case = AndTestCase(
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
            
    test_writer.add_tests(test_cases)
    test_writer.save()
    
