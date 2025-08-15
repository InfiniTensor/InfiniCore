from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor
import torch
from enum import Enum, auto
class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]

def rms_norm_backward(
    grad_y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    inplace: Inplace
):
    torch_grad_y = torch.from_numpy(grad_y)
    torch_x = torch.from_numpy(x)
    torch_w = torch.from_numpy(w)
    torch_x.requires_grad_(True)

    rmsNorm = torch.nn.RMSNorm(normalized_shape=[torch_x.shape[-1]], eps=0, dtype=torch.float64)
    rmsNorm.weight.data = torch_w
    torch_y = rmsNorm(torch_x)
    torch_y.backward(torch_grad_y)

    return torch_x.grad.detach().numpy(),\
            rmsNorm.weight.grad.detach().numpy()


class RMSNormBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        grad_x: np.ndarray,
        grad_x_shape: List[int],
        grad_x_strides: List[int],
        grad_w: np.ndarray,
        grad_w_shape: List[int],
        grad_w_strides: List[int],
        grad_y: np.ndarray,
        grad_y_shape: List[int],
        grad_y_strides: List[int],
        x: np.ndarray,
        x_shape: List[int],
        x_strides: List[int],
        w: np.ndarray,
        w_shape: List[int],
        w_strides: List[int],
        inplace: Inplace
    ):
        super().__init__("rms_norm_backward")
        self.grad_x = grad_x
        self.grad_x_shape = grad_x_shape
        self.grad_x_strides = grad_x_strides
        self.grad_w = grad_w
        self.grad_w_shape = grad_w_shape
        self.grad_w_strides = grad_w_strides
        self.grad_y = grad_y
        self.grad_y_shape = grad_y_shape
        self.grad_y_strides = grad_y_strides
        self.x = x
        self.x_shape = x_shape
        self.x_strides = x_strides
        self.w = w
        self.w_shape = w_shape
        self.w_strides = w_strides
        self.inplace = inplace

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("grad_x.shape"), self.grad_x_shape)
        test_writer.add_array(test_writer.gguf_key("grad_w.shape"), self.grad_w_shape)
        test_writer.add_array(test_writer.gguf_key("grad_y.shape"), self.grad_y_shape)
        test_writer.add_array(test_writer.gguf_key("x.shape"), self.x_shape)
        test_writer.add_array(test_writer.gguf_key("w.shape"), self.w_shape)
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
        test_writer.add_tensor(
            test_writer.gguf_key("grad_x"), self.grad_x, raw_dtype=np_dtype_to_ggml(self.grad_x.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_w"), self.grad_w, raw_dtype=np_dtype_to_ggml(self.grad_w.dtype)
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

        ans_grad_x, ans_grad_w = rms_norm_backward(
            self.grad_y.astype(np.float64),
            self.x.astype(np.float64),
            self.w.astype(np.float64),
            self.inplace
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_x"), ans_grad_x, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_w"), ans_grad_w, raw_dtype=gguf.GGMLQuantizationType.F64
        )
if __name__ == "__main__":
    test_writer = InfiniopTestWriter("rms_norm_backward.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, grad_x_strides, grad_w_strides, grad_y_strides, x_strides
        ([13, 4, 5], [37, 5, 1], [2], [38, 5, 1], [39, 5, 1]),
        ([20, 30, 40], [1555, 40, 1], None, None, None),
        ([55, 65, 10], None, [10], None, None),
        ([155, 165, 110], None, None, [40037, 110, 1], None),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for shape, grad_x_strides, grad_w_strides, grad_y_strides, x_strides in _TEST_CASES_:
            for inplace in _INPLACE:
                grad_y = np.random.rand(*shape).astype(dtype)
                x = np.random.rand(*shape).astype(dtype)
                w = np.random.rand(shape[-1]).astype(dtype)
                if inplace == Inplace.INPLACE:
                    grad_x = grad_y
                else:
                    grad_x = np.empty(shape, dtype=dtype)             
                grad_w = np.empty([shape[-1]], dtype=dtype)

                test_case = RMSNormBackwardTestCase(
                    grad_x=grad_x,
                    grad_x_shape=shape,
                    grad_x_strides=grad_x_strides,
                    grad_w=grad_w,
                    grad_w_shape=[shape[-1]],
                    grad_w_strides=grad_w_strides,
                    grad_y=grad_y,
                    grad_y_shape=shape,
                    grad_y_strides=grad_y_strides,
                    x=x,
                    x_shape=shape,
                    x_strides=x_strides,
                    w=w,
                    w_shape=[shape[-1]],
                    w_strides=None,
                    inplace=inplace
                )
                test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
