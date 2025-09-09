from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor
import torch

def linear(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    bias_exist: bool
):
    torch_x = torch.from_numpy(x)
    torch_w = torch.from_numpy(w)
    if bias_exist:
        torch_b = torch.from_numpy(b)
    return torch.nn.functional.linear(torch_x, torch_w, bias=(torch_b if bias_exist else None)).detach().numpy()


class LinearTestCase(InfiniopTestCase):
    def __init__(
        self,
        y: np.ndarray,
        y_shape: List[int],
        y_strides: List[int],
        x: np.ndarray,
        x_shape: List[int],
        x_strides: List[int],
        w: np.ndarray,
        w_shape: List[int],
        w_strides: List[int],
        b: np.ndarray,
        b_shape: List[int],
        b_strides: List[int],
        bias_exist: bool
    ):
        super().__init__("linear")
        self.y = y
        self.y_shape = y_shape
        self.y_strides = y_strides
        self.x = x
        self.x_shape = x_shape
        self.x_strides = x_strides
        self.w = w
        self.w_shape = w_shape
        self.w_strides = w_strides
        self.b = b
        self.b_shape = b_shape
        self.b_strides = b_strides
        self.bias_exist = bias_exist

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("y.shape"), self.y_shape)
        test_writer.add_array(test_writer.gguf_key("x.shape"), self.x_shape)
        test_writer.add_array(test_writer.gguf_key("w.shape"), self.w_shape)
        test_writer.add_array(test_writer.gguf_key("b.shape"), self.b_shape)
        test_writer.add_bool(test_writer.gguf_key("bias_exist"), self.bias_exist)
        if self.x_strides is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*self.x_strides))
        if self.w_strides is not None:
            test_writer.add_array(test_writer.gguf_key("w.strides"), gguf_strides(*self.w_strides))
        if self.b_strides is not None:
            test_writer.add_array(test_writer.gguf_key("b.strides"), gguf_strides(*self.b_strides))
        test_writer.add_array(
            test_writer.gguf_key("y.strides"),
            gguf_strides(*(self.y_strides if self.y_strides is not None else contiguous_gguf_strides(self.y_shape)))
        )
        test_writer.add_tensor(
            test_writer.gguf_key("y"), self.y, raw_dtype=np_dtype_to_ggml(self.y.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("x"), self.x, raw_dtype=np_dtype_to_ggml(self.x.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("w"), self.w, raw_dtype=np_dtype_to_ggml(self.w.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("b"), self.b, raw_dtype=np_dtype_to_ggml(self.y.dtype)
        )

        ans = linear(
            self.x.astype(np.float64),
            self.w.astype(np.float64),
            self.b.astype(np.float64),
            self.bias_exist
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )
if __name__ == "__main__":
    test_writer = InfiniopTestWriter("linear.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # in_features, out_features, bias_exist, x_strides, y_strides, w_strides
        (50, 40, True, None, None, [1, 377]),
        (50, 40, False, [10], [1], None),
        (50, 40, True, [10], [1], None),
        (333, 999, True, [1], [10], None),
        (333, 999, False, [1], [10], None),        
        (1001, 505, True, None, None, [3001, 3]),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for in_features, out_features, bias_exist, x_strides, y_strides, w_strides in _TEST_CASES_:
            x = np.random.rand(in_features).astype(dtype)
            w = np.random.rand(out_features, in_features).astype(dtype)
            if bias_exist:
                b = np.random.rand(out_features).astype(dtype)
            else:
                b = np.empty(shape=[], dtype=dtype)
            y = np.empty(out_features, dtype=dtype)

            test_case = LinearTestCase(
                y=y,
                y_shape=[out_features],
                y_strides=y_strides,
                x=x,
                x_shape=[in_features],
                x_strides=x_strides,
                w=w,
                w_shape=[out_features, in_features],
                w_strides=w_strides,
                b=b,
                b_shape=b.shape,
                b_strides=None,
                bias_exist=bias_exist
            )
            test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
