import torch
import gguf
import numpy as np
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def random_tensor(shape: List[int], dtype: torch.dtype) -> torch.Tensor:
    torch_supported = dtype in [torch.int32, torch.int64, torch.uint8, torch.float16, torch.float32, torch.float64]

    if torch_supported:
        if dtype in [torch.int32, torch.int64]:
            return torch.randint(-100, 100, shape, dtype=dtype)
        elif dtype == torch.uint8:
            return torch.randint(0, 200, shape, dtype=dtype)
        elif dtype in [torch.float16, torch.float32, torch.float64]:
            return torch.rand(*shape, dtype=dtype) * 10 - 5
    else:
        np_dtype_map = {
            torch.uint32: np.uint32,
            torch.uint64: np.uint64,
        }
        np_dtype = np_dtype_map.get(dtype)
        if np.issubdtype(np_dtype, np.unsignedinteger):
            np_array = np.random.randint(0, 200, size=shape, dtype=np_dtype)
        else:
            np_array = np.random.randint(-100, 100, size=shape, dtype=np_dtype)

        return torch.from_numpy(np_array)

_INT_DTYPES_ = [torch.int32, torch.int64]
_UINT_DTYPES_ = [torch.uint8, torch.uint32, torch.uint64]
_FLOAT_DTYPES_ = [torch.float16, torch.float32, torch.float64]
_ALL_SOURCE_DTYPES_ = _INT_DTYPES_ + _UINT_DTYPES_ + _FLOAT_DTYPES_

def is_valid_cast(src: torch.dtype, dst: torch.dtype) -> bool:
    if src == dst:
        return False
    if src in _INT_DTYPES_ + _UINT_DTYPES_ and dst in _INT_DTYPES_ + _UINT_DTYPES_:
        return True
    if src in _FLOAT_DTYPES_ and dst in _FLOAT_DTYPES_:
        return True
    if src in _INT_DTYPES_ + _UINT_DTYPES_ and dst in _FLOAT_DTYPES_:
        return True
    return False

class CastTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_tensor: torch.Tensor,
        shape: List[int] | None,
        stride: List[int] | None,
    ):
        super().__init__("cast")
        self.input = input_tensor
        self.shape = shape
        self.stride = stride

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape)
        test_writer.add_array(
            test_writer.gguf_key("input.strides"),
            gguf_strides(*self.stride if self.stride is not None else contiguous_gguf_strides(self.shape)),
        )
        np_input = self.input.numpy()
        dtype = np_input.dtype

        if dtype == np.uint8:
            np_input = np_input.astype(np.int16)
        elif dtype == np.uint32:
            np_input = np_input.astype(np.int64)
        elif dtype == np.uint64:
            np_input = np_input.astype(np.float64)
        # GGMLQuantizationType Only F16, F32, F64, I8, I16, I32, I64 tensors are supported for now
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            np_input,
            raw_dtype=np_dtype_to_ggml(np_input.dtype),
        )

        src_dtype = self.input.dtype
        for dst_dtype in _ALL_SOURCE_DTYPES_:
            if not is_valid_cast(src_dtype, dst_dtype):
                continue

            output_tensor = self.input.to(dst_dtype)
            np_output = output_tensor.numpy()
            out_dtype = np_output.dtype

            if out_dtype == np.uint8:
                np_output = np_output.astype(np.int16)
            elif out_dtype == np.uint32:
                np_output = np_output.astype(np.int64)
            elif out_dtype == np.uint64:
                np_output = np_output.astype(np.float64)
            # GGMLQuantizationType Only F16, F32, F64, I8, I16, I32, I64 tensors are supported for now
            test_writer.add_tensor(
                test_writer.gguf_key(f"output.to_{str(dst_dtype).split('.')[-1]}"),
                np_output,
                raw_dtype=np_dtype_to_ggml(np_output.dtype),
            )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("cast.gguf")
    test_cases: List[CastTestCase] = []

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

    for shape, stride in _TEST_CASES_:
        for src_dtype in _ALL_SOURCE_DTYPES_:
            input_tensor = random_tensor(shape, src_dtype)
            
            test_case = CastTestCase(
                input_tensor=input_tensor,
                shape=list(shape),
                stride=list(stride) if stride is not None else None,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
