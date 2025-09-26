from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def ones(x: np.ndarray):
    return np.ones_like(x)


class OnesTestCase(InfiniopTestCase):
    def __init__(self,
                 x: np.ndarray,
                 shape_x: List[int] | None,
                 stride_x: List[int] | None,
                 y: np.ndarray,
                 shape_y: List[int] | None,
                 stride_y: List[int] | None
                 ):
        super().__init__("ones")
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
        ans = ones(
            self.x.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("ones.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, x_stride, y_stride
        ((13, 4), None, None),
        ((13, 4), (10, 1), (10, 1)),
        ((13, 4, 4), None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
        ((16, 5632), None, None),
        ((16, 5632), (13312, 1), (13312, 1)),
        ((4, 4, 5632), None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
    ]

    _TENSOR_DTYPES_ = [np.bool_,  # 2
                       np.int8,  # 3
                       np.int16,  # 4
                       np.int32,  # 5
                       np.int64,  # 6
                       # np.uint8,  # 7
                       # np.uint16,  # 8
                       # np.uint32,  # 9
                       # np.uint64,  # 10
                       # InfiniDtype.F8,  # 11
                       np.float16,  # 12
                       np.float32,  # 13
                       np.float64,  # 14
                       # InfiniDtype.BF16,  # 19
                       ]

    for dtype in _TENSOR_DTYPES_:
        for shape, stride_x, stride_y in _TEST_CASES_:
            x = np.random.rand(*shape).astype(dtype)

            y = np.empty(tuple(0 for _ in shape), dtype=dtype)
            x = process_zero_stride_tensor(x, stride_x)

            test_case = OnesTestCase(
                x=x,
                shape_x=shape,
                stride_x=stride_x,
                y=y,
                shape_y=shape,
                stride_y=stride_y,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
