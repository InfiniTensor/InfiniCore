import numpy as np
import torch
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides


def reference_exp(input: torch.Tensor) -> torch.Tensor:
    return torch.exp(input)


class ExpTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: torch.Tensor,
        shape: List[int] | None,
        stride: List[int] | None,
    ):
        super().__init__("exp")
        self.input = input
        self.shape = shape
        self.stride = stride

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        # 添加input的形状和步幅
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape)
        strides = self.stride if self.stride is not None else contiguous_gguf_strides(self.shape)
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*strides))
        
        # 添加output的形状和步幅（与input相同）
        test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape)
        # 确保output使用连续的步幅
        test_writer.add_array(test_writer.gguf_key("output.strides"), gguf_strides(*contiguous_gguf_strides(self.shape)))
        
        # 处理输入张量
        if self.input.dtype == torch.bfloat16:
            input_numpy = self.input.view(torch.uint16).numpy()
            ggml_dtype = gguf.GGMLQuantizationType.BF16
        else:
            input_numpy = self.input.numpy()
            ggml_dtype = np_dtype_to_ggml(input_numpy.dtype)
        
        # 添加input张量
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=ggml_dtype,
        )
        
        # 创建output张量（输出缓冲区）
        # 确保输出类型与输入类型一致
        output_numpy = np.empty_like(input_numpy)
        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            output_numpy,
            raw_dtype=ggml_dtype,
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("exp.gguf")
    test_cases: List[ExpTestCase] = []

    _TEST_CASES_ = [
        ((3, 3), None),
        ((32, 512), None),
        ((32, 512), (1024, 1)),
        ((4, 4, 4), None),
        ((16, 32, 512), None),
        ((16, 20, 512), (20480, 512, 1)),
        ((1024,), None),
        ((1024,), (2,)),
        ((2, 3, 4, 5), None),
    ]

    _TENSOR_DTYPES_ = [torch.float16, torch.float32, torch.bfloat16]

    for dtype in _TENSOR_DTYPES_:
        for shape, stride in _TEST_CASES_:
            # 生成小范围的随机数，避免exp溢出
            input_tensor = torch.rand(*shape, dtype=dtype) * 4 - 2

            test_case = ExpTestCase(
                input_tensor,
                list(shape),
                list(stride) if stride is not None else None,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
