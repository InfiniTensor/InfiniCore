import torch
from ast import List
import numpy as np
import gguf
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, tensor_to_numpy

class LinearBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor | None,
        grad_y: torch.Tensor,
        grad_x: torch.Tensor,
        grad_w: torch.Tensor,
        grad_b: torch.Tensor | None,
        x_shape: List[int] | None,
        w_shape: List[int] | None,
        b_shape: List[int] | None, 
        grad_y_shape: List[int] | None,
        x_strides: List[int] | None,
        w_strides: List[int] | None,
        b_strides: List[int] | None,
        grad_y_strides: List[int] | None,
        grad_x_strides: List[int] | None,
        grad_w_strides: List[int] | None,
        grad_b_strides: List[int] | None,
    ):
        super().__init__("linear_backward")
        self.x = x
        self.w = w
        self.b = b
        self.grad_y = grad_y
        self.grad_x = grad_x
        self.grad_w = grad_w
        self.grad_b = grad_b
        self.x_shape = x_shape
        self.w_shape = w_shape
        self.b_shape = b_shape
        self.grad_y_shape = grad_y_shape
        self.x_strides = x_strides or contiguous_gguf_strides(x.shape)
        self.w_strides = w_strides or contiguous_gguf_strides(w.shape)
        self.b_strides = b_strides or contiguous_gguf_strides(b.shape) if b is not None else None
        self.grad_y_strides = grad_y_strides or contiguous_gguf_strides(grad_y.shape)
        self.grad_x_strides = grad_x_strides or contiguous_gguf_strides(x.shape)
        self.grad_w_strides = grad_w_strides or contiguous_gguf_strides(w.shape)
        self.grad_b_strides = grad_b_strides or contiguous_gguf_strides(b.shape) if b is not None else None

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # 写入输入数据 x
        test_writer.add_array(test_writer.gguf_key("x.shape"), self.x_shape)
        test_writer.add_array(test_writer.gguf_key("x.strides"), self.x_strides)
        # x_numpy = self.x.detach().cpu().numpy()
        x_numpy = tensor_to_numpy(self.x)
        test_writer.add_tensor(test_writer.gguf_key("x"), x_numpy, raw_dtype=np_dtype_to_ggml(x_numpy.dtype))

        # 写入权重 w
        test_writer.add_array(test_writer.gguf_key("w.shape"), self.w_shape)
        test_writer.add_array(test_writer.gguf_key("w.strides"), self.w_strides)
        # w_numpy = self.w.detach().cpu().numpy()
        w_numpy = tensor_to_numpy(self.w)
        test_writer.add_tensor(test_writer.gguf_key("w"), w_numpy, raw_dtype=np_dtype_to_ggml(w_numpy.dtype))

        # 写入偏置 b
        if self.b is not None:
            test_writer.add_array(test_writer.gguf_key("b.shape"), self.b_shape)
            test_writer.add_array(test_writer.gguf_key("b.strides"), self.b_strides)
            # b_numpy = self.b.detach().cpu().numpy()
            b_numpy = tensor_to_numpy(self.b)
            test_writer.add_tensor(test_writer.gguf_key("b"), b_numpy, raw_dtype=np_dtype_to_ggml(b_numpy.dtype))

        # 写入输出梯度 grad_y
        test_writer.add_array(test_writer.gguf_key("grad_y.shape"), self.grad_y_shape)
        test_writer.add_array(test_writer.gguf_key("grad_y.strides"), self.grad_y_strides)
        # grad_y_numpy = self.grad_y.detach().cpu().numpy()
        grad_y_numpy = tensor_to_numpy(self.grad_y)
        test_writer.add_tensor(test_writer.gguf_key("grad_y"), grad_y_numpy, raw_dtype=np_dtype_to_ggml(grad_y_numpy.dtype))

        # 写入grad_x, grad_w, grad_b的形状和步长和数据
        test_writer.add_array(test_writer.gguf_key("grad_x.shape"), self.x_shape)
        test_writer.add_array(test_writer.gguf_key("grad_x.strides"), self.grad_x_strides)
        # grad_x_numpy = self.grad_x.detach().cpu().numpy()
        grad_x_numpy = tensor_to_numpy(self.grad_x)
        test_writer.add_tensor(
            test_writer.gguf_key("grad_x"), grad_x_numpy, raw_dtype=np_dtype_to_ggml(grad_x_numpy.dtype)
        )
        test_writer.add_array(test_writer.gguf_key("grad_w.shape"), self.w_shape)
        test_writer.add_array(test_writer.gguf_key("grad_w.strides"), self.grad_w_strides)
        # grad_w_numpy = self.grad_w.detach().cpu().numpy()
        grad_w_numpy = tensor_to_numpy(self.grad_w)
        test_writer.add_tensor(
            test_writer.gguf_key("grad_w"), grad_w_numpy, raw_dtype=np_dtype_to_ggml(grad_w_numpy.dtype)
        )
        if self.b is not None:
            test_writer.add_array(test_writer.gguf_key("grad_b.shape"), self.b_shape)
            if self.grad_b_strides is not None:
                test_writer.add_array(test_writer.gguf_key("grad_b.strides"), self.grad_b_strides)
            # grad_b_numpy = self.grad_b.detach().cpu().numpy()
            grad_b_numpy = tensor_to_numpy(self.grad_b)
            test_writer.add_tensor(
                test_writer.gguf_key("grad_b"), grad_b_numpy, raw_dtype=np_dtype_to_ggml(grad_b_numpy.dtype)
            )

        # 启用梯度计算 - 转换为FP64进行高精度计算
        ans_x = self.x.to(torch.float64).detach().requires_grad_(True)
        ans_w = self.w.to(torch.float64).detach().requires_grad_(True)
        ans_b = self.b.to(torch.float64).detach().requires_grad_(True) if self.b is not None else None

        # 将梯度输出也转换为FP64
        grad_y_fp64 = self.grad_y.to(torch.float64)

        # 前向传播
        y = torch.nn.functional.linear(ans_x, ans_w, ans_b)

        # 反向传播
        torch.autograd.backward(y, grad_tensors=grad_y_fp64, retain_graph=True)

        # 写入计算得到的梯度
        # grad_x (输入梯度)
        ans_x_grad = ans_x.grad
        # ans_x_grad_numpy = ans_x_grad.detach().cpu().numpy()
        ans_x_grad_numpy = ans_x_grad.detach().cpu().numpy()
        test_writer.add_tensor(test_writer.gguf_key("ans_grad_x"), ans_x_grad_numpy, raw_dtype=gguf.GGMLQuantizationType.F64)

        # grad_w (权重梯度)
        ans_w_grad = ans_w.grad
        # ans_w_grad_numpy = ans_w_grad.detach().cpu().numpy()
        ans_w_grad_numpy = ans_w_grad.detach().cpu().numpy()
        test_writer.add_tensor(test_writer.gguf_key("ans_grad_w"), ans_w_grad_numpy, raw_dtype=gguf.GGMLQuantizationType.F64)

        # grad_b (偏置梯度)
        if ans_b is not None:
            ans_b_grad = ans_b.grad
            # ans_b_grad_numpy = ans_b_grad.detach().cpu().numpy()
            ans_b_grad_numpy = ans_b_grad.detach().cpu().numpy()
            test_writer.add_tensor(test_writer.gguf_key("ans_grad_b"), ans_b_grad_numpy, raw_dtype=gguf.GGMLQuantizationType.F64)


def gen_gguf(dtype: torch.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

# (x_shape, w_shape, b_shape, grad_y_shape,
#  x_strides, w_strides, b_strides, grad_y_strides,
#  grad_x_strides, grad_w_strides, grad_b_strides)
    _TEST_CASES_ = [
        # 基本测试
        ((3, 4), (2, 4), (2,), (3, 2), None, None, None, None, None, None, None),
        ((5, 4), (3, 4), (3,), (5, 3), None, None, None, None, None, None, None),

        # 多维输入
        ((2, 3, 4), (5, 4), (5,), (2, 3, 5), None, None, None, None, None, None, None),
        ((1, 2, 3, 4), (6, 4), (6,), (1, 2, 3, 6), None, None, None, None, None, None, None),

        # 步长测试
        ((4, 4), (3, 4), (3,), (4, 3), [8, 2], [6, 3], [3], [6, 2], [4, 1], [3, 1], [1]),
        ((2, 3, 4), (5, 4), (5,), (2, 3, 5), [24, 8, 2], [8, 2], [5], [30, 10, 2], [24, 8, 2], [8, 2], [5]),

        # batch=1
        ((1, 3), (2, 3), (2,), (1, 2), None, None, None, None, None, None, None),

        # 大特征
        ((6, 1024), (128, 1024), (128,), (6, 128), None, None, None, None, None, None, None),

        # b为0/None（无偏置）
        ((4, 8), (6, 8), None, (4, 6), None, None, None, None, None, None, None),
        ((2, 3, 5), (7, 5), None, (2, 3, 7), [30, 10, 2], [5, 1], None, [21, 3, 1], [30, 10, 2], [5, 1], None),
    ]

    for x_shape, w_shape, b_shape, grad_y_shape, x_strides, w_strides, b_strides, grad_y_strides, grad_x_strides, grad_w_strides, grad_b_strides in _TEST_CASES_:
        # 生成输入、权重和输出梯度
        if dtype == bfloat16:  
            dtype = torch.bfloat16
        x = torch.rand(*x_shape, dtype=dtype, requires_grad=True)
        w = torch.rand(*w_shape, dtype=dtype, requires_grad=True)
        b = torch.rand(*b_shape, dtype=dtype, requires_grad=True) if b_shape is not None else None
        grad_y = torch.rand(*grad_y_shape, dtype=dtype)
        grad_x = torch.empty(x_shape, dtype=dtype)
        grad_w = torch.empty(w_shape, dtype=dtype)
        grad_b = torch.empty(b_shape, dtype=dtype) if b_shape is not None else None

        test_case = LinearBackwardTestCase(
            x=x,
            w=w,
            b=b,
            grad_y=grad_y,
            grad_x=grad_x,
            grad_w=grad_w,
            grad_b=grad_b,
            x_shape=x_shape,
            w_shape=w_shape,
            b_shape=b_shape,
            grad_y_shape=grad_y_shape,
            x_strides=x_strides,
            w_strides=w_strides,
            b_strides=b_strides,
            grad_y_strides=grad_y_strides,
            grad_x_strides=grad_x_strides,
            grad_w_strides=grad_w_strides,
            grad_b_strides=grad_b_strides,
        )
        test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()

if __name__ == "__main__":
    _TENSOR_DTYPES_ = [
        bfloat16,
        torch.float32,
        torch.float16,
    ]

    dtype_filename_map = {
        bfloat16: "linear_backward_bf16.gguf",
        torch.float32: "linear_backward_f32.gguf",
        torch.float16: "linear_backward_f16.gguf",
    }   
    
    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)