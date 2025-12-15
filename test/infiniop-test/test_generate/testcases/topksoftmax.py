import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides


def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, shape).astype(dtype) * 0.001


def torch_Topksoftmax(router_logits, top_k: int, norm_topk_prob: bool):
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(torch.float)
    return routing_weights, selected_experts


def python_Topksoftmax(router_logits, top_k: int, norm_topk_prob: bool):
    router_logits = torch.from_numpy(router_logits)
    lable_values, lable_indices = torch_Topksoftmax(router_logits, top_k, norm_topk_prob)
    return lable_values.numpy(), lable_indices.numpy()


class TopksoftmaxTestCase(InfiniopTestCase):
    def __init__(self,
                 values: np.ndarray,  # 传出参数
                 indices: np.ndarray,  # 传出参数
                 x: np.ndarray,  # 传入参数
                 topk: np.ndarray,
                 norm: bool,
                 values_shape: List[int] | None,
                 values_strides: List[int] | None,
                 indices_shape: List[int] | None,
                 indices_strides: List[int] | None,
                 x_shape: List[int] | None,
                 x_strides: List[int] | None,
                 ):
        super().__init__("topksoftmax")
        self.values = values
        self.indices = indices
        self.x = x
        self.topk = topk
        self.norm = norm

        self.values_shape = values_shape
        self.values_strides = values_strides
        self.indices_shape = indices_shape
        self.indices_strides = indices_strides
        self.x_shape = x_shape
        self.x_strides = x_strides

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        if self.values_shape is not None:
            print("self.values_shape:  ", self.values_shape)
            test_writer.add_array(test_writer.gguf_key("values.shape"), self.values_shape)
        if self.indices_shape is not None:
            test_writer.add_array(test_writer.gguf_key("indices.shape"), self.indices_shape)
        if self.x_shape is not None:
            test_writer.add_array(test_writer.gguf_key("x.shape"), self.x_shape)

        if self.x_strides is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*self.x_strides))

        test_writer.add_array(
            test_writer.gguf_key("values.strides"),
            gguf_strides(*self.values_strides if self.values_strides is not None else contiguous_gguf_strides(self.values_shape))
        )

        test_writer.add_array(
            test_writer.gguf_key("indices.strides"),
            gguf_strides(*self.indices_strides if self.indices_strides is not None else contiguous_gguf_strides(self.indices_shape))
        )

        test_writer.add_tensor(test_writer.gguf_key("values"),
                               self.values,
                               raw_dtype=np_dtype_to_ggml(self.values.dtype))

        test_writer.add_tensor(test_writer.gguf_key("indices"),
                               self.indices,
                               raw_dtype=np_dtype_to_ggml(self.indices.dtype))

        test_writer.add_tensor(test_writer.gguf_key("x"),
                               self.x,
                               raw_dtype=np_dtype_to_ggml(self.x.dtype))

        test_writer.add_int32(test_writer.gguf_key("topk"), self.topk)
        test_writer.add_bool(test_writer.gguf_key("norm"), self.norm)

        lable_values, lable_indices = python_Topksoftmax(self.x.copy(), self.topk, self.norm)

        test_writer.add_tensor(
            test_writer.gguf_key("lable_values"),
            lable_values,
            raw_dtype=np_dtype_to_ggml(lable_values.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("lable_indices"),
            lable_indices,
            raw_dtype=np_dtype_to_ggml(lable_indices.dtype)
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("topksoftmax.gguf")
    test_cases = []

    _TEST_CASES_ = [
        # x_shape, x_strides, topk, norm
        ((1, 32), None, 4, True),
        ((8, 20), None, 8, False),
        ((2, 128), None, 10, True)
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for x_shape, x_strides, topk, norm in _TEST_CASES_:
            ntoken = x_shape[0]

            values_indices_shape = (ntoken, topk)
            values = np.empty(tuple(0 for _ in values_indices_shape), dtype=np.float32)
            indices = np.empty(tuple(0 for _ in values_indices_shape), dtype=np.int32)

            x = np.random.rand(*x_shape).astype(dtype)

            test_case = TopksoftmaxTestCase(
                values=values,
                indices=indices,
                x=x,
                topk=topk,
                norm=norm,
                values_shape=list(values_indices_shape),
                values_strides=None,
                indices_shape=list(values_indices_shape),
                indices_strides=None,
                x_shape=list(x_shape),
                x_strides=None
            )

            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
