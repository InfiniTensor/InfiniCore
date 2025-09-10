from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor
import torch

def linear_backward(
    grad_y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    bias_exist: bool
):
    torch_grad_y = torch.from_numpy(grad_y)
    torch_x = torch.from_numpy(x)
    torch_w = torch.from_numpy(w)

    torch_x.requires_grad_(True)
    torch_w.requires_grad_(True)
    if bias_exist:
        torch_b = torch.from_numpy(b)
        torch_b.requires_grad_(True)

    torch_y = torch.nn.functional.linear(torch_x, torch_w, bias=(torch_b if bias_exist else None))
    torch_y.backward(torch_grad_y)
    return torch_x.grad.detach().numpy(), \
            torch_w.grad.detach().numpy(), \
            torch_b.grad.detach().numpy() if bias_exist else np.empty([], dtype=np.float64)


class LinearBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        grad_x: np.ndarray,
        grad_x_shape: List[int],
        grad_x_strides: List[int],
        grad_w: np.ndarray,
        grad_w_shape: List[int],
        grad_w_strides: List[int],
        grad_b: np.ndarray,
        grad_b_shape: List[int],
        grad_b_strides: List[int],
        grad_y: np.ndarray,
        grad_y_shape: List[int],
        grad_y_strides: List[int],
        x: np.ndarray,
        x_shape: List[int],
        x_strides: List[int],
        w: np.ndarray,
        w_shape: List[int],
        w_strides: List[int],
        b: np.ndarray,
        bias_exist: bool
    ):
        super().__init__("linear_backward")
        self.grad_x = grad_x
        self.grad_x_shape = grad_x_shape
        self.grad_x_strides = grad_x_strides
        self.grad_w = grad_w
        self.grad_w_shape = grad_w_shape
        self.grad_w_strides = grad_w_strides
        self.grad_b = grad_b
        self.grad_b_shape = grad_b_shape
        self.grad_b_strides = grad_b_strides
        self.grad_y = grad_y
        self.grad_y_shape = grad_y_shape
        self.grad_y_strides = grad_y_strides
        self.x = x
        self.x_shape = x_shape
        self.x_strides = x_strides
        self.w = w
        self.w_shape = w_shape
        self.w_strides = w_strides

        self.b = b
        self.bias_exist = bias_exist

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("grad_x.shape"), self.grad_x_shape)
        test_writer.add_array(test_writer.gguf_key("grad_w.shape"), self.grad_w_shape)
        test_writer.add_array(test_writer.gguf_key("grad_b.shape"), self.grad_b_shape)
        test_writer.add_array(test_writer.gguf_key("grad_y.shape"), self.grad_y_shape)
        test_writer.add_array(test_writer.gguf_key("x.shape"), self.x_shape)
        test_writer.add_array(test_writer.gguf_key("w.shape"), self.w_shape)
        test_writer.add_bool(test_writer.gguf_key("bias_exist"), self.bias_exist)
        if self.grad_y_strides is not None:
            test_writer.add_array(test_writer.gguf_key("grad_y.strides"), gguf_strides(*self.grad_y_strides))
        if self.x_strides is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*self.x_strides))
        if self.w_strides is not None:
            test_writer.add_array(test_writer.gguf_key("w.strides"), gguf_strides(*self.w_strides))
        test_writer.add_array(
            test_writer.gguf_key("grad_x.strides"),
            gguf_strides(*(self.grad_x_strides if self.grad_x_strides is not None else contiguous_gguf_strides(self.grad_x_shape)))
        )
        test_writer.add_array(
            test_writer.gguf_key("grad_w.strides"),
            gguf_strides(*(self.grad_w_strides if self.grad_w_strides is not None else contiguous_gguf_strides(self.grad_w_shape)))
        )
        test_writer.add_array(
            test_writer.gguf_key("grad_b.strides"),
            gguf_strides(*(self.grad_b_strides if self.grad_b_strides is not None else contiguous_gguf_strides(self.grad_b_shape)))
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_x"), self.grad_x, raw_dtype=np_dtype_to_ggml(self.grad_x.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_w"), self.grad_w, raw_dtype=np_dtype_to_ggml(self.grad_w.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_b"), self.grad_b, raw_dtype=np_dtype_to_ggml(self.grad_b.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_y"), self.grad_y, raw_dtype=np_dtype_to_ggml(self.grad_y.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("x"), self.x, raw_dtype=np_dtype_to_ggml(self.x.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("w"), self.w, raw_dtype=np_dtype_to_ggml(self.w.dtype)
        )

        ans_grad_x, ans_grad_w, ans_grad_b = linear_backward(
            self.grad_y.astype(np.float64),
            self.x.astype(np.float64),
            self.w.astype(np.float64),
            self.b.astype(np.float64),
            self.bias_exist
        )

        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_x"), ans_grad_x, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_w"), ans_grad_w, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_b"), ans_grad_b, raw_dtype=gguf.GGMLQuantizationType.F64
        )
if __name__ == "__main__":
    test_writer = InfiniopTestWriter("linear_backward.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # in_features, out_features, bias_exist, grad_x_strides, grad_y_strides, grad_w_strides
        (50, 40, True, None, None, [1, 377]),
        (50, 40, False, [10], [1], None),
        (50, 40, True, [10], [1], None),
        (333, 999, True, [1], [10], None),
        (333, 999, False, [1], [10], None),      
        (1001, 505, True, None, None, [3001, 3]),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for in_features, out_features, bias_exist, grad_x_strides, grad_y_strides, grad_w_strides in _TEST_CASES_:
            grad_y = np.random.rand(out_features).astype(dtype)
            x = np.random.rand(in_features).astype(dtype)
            w = np.random.rand(out_features, in_features).astype(dtype)
            b = np.random.rand(out_features).astype(dtype) if bias_exist else \
                np.empty([], dtype=dtype)
            grad_x = np.empty([in_features], dtype=dtype)
            grad_w = np.empty([out_features, in_features], dtype=dtype)
            grad_b = np.empty([out_features], dtype=dtype)

            test_case = LinearBackwardTestCase(
                grad_x=grad_x,
                grad_x_shape=[in_features],
                grad_x_strides=grad_x_strides,
                grad_w=grad_w,
                grad_w_shape=[out_features, in_features],
                grad_w_strides=grad_w_strides,
                grad_b=grad_b,
                grad_b_shape=[out_features],
                grad_b_strides=None,
                grad_y=grad_y,
                grad_y_shape=[out_features],
                grad_y_strides=grad_y_strides,
                x=x,
                x_shape=[in_features],
                x_strides=None,
                w=w,
                w_shape=[out_features, in_features],
                w_strides=None,
                b=b,
                bias_exist=bias_exist
            )
            test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
