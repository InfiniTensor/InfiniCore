# test_cast_gguf.py
import torch
import gguf
import numpy as np
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def random_tensor(shape: List[int], dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.int8:
        low, high = -128, 127
        return torch.randint(low, high, size=shape, dtype=dtype)
    elif dtype == torch.int16:
        low, high = -32768, 32767
        return torch.randint(low, high, size=shape, dtype=dtype)
    elif dtype in (torch.int32, torch.int64):
        low, high = -2000, 2000
        return torch.randint(low, high, size=shape, dtype=dtype)
    elif dtype == torch.bool:
        return torch.randint(0, 2, size=shape, dtype=torch.bool)
    elif dtype in (torch.float16, torch.bfloat16):
        t = (torch.rand(*shape, dtype=torch.float32) * 10) - 5
        return t.to(dtype)
    elif dtype == torch.float32:
        return (torch.rand(*shape, dtype=torch.float32) * 10) - 5
    elif dtype == torch.float64:
        return (torch.rand(*shape, dtype=torch.float64) * 10) - 5
    if dtype == torch.uint8:
        return (torch.randint(0, 256, size=shape, dtype=torch.uint8))
    if getattr(torch, "uint32", None) is not None and dtype == getattr(torch, "uint32"):
        return (torch.randint(0, 2000, size=shape, dtype=torch.int64).to(dtype))
    if getattr(torch, "uint64", None) is not None and dtype == getattr(torch, "uint64"):
        return (torch.randint(0, 2000, size=shape, dtype=torch.int64).to(dtype))
    return (torch.rand(*shape, dtype=torch.float32) * 10) - 5

_INT_DTYPES_ = [torch.int32, torch.int64]
_UINT_DTYPES_ = [getattr(torch, "uint32", None), getattr(torch, "uint64", None)]
_FLOAT_DTYPES_ = [torch.float16, torch.float32, torch.float64]

_ALL_DTYPES_ = [dt for dt in (_INT_DTYPES_ + _UINT_DTYPES_ + _FLOAT_DTYPES_) if dt is not None]

def is_valid_cast(src: torch.dtype, dst: torch.dtype) -> bool:
    if src == dst:
        return False

    is_src_int = src in _INT_DTYPES_
    is_dst_int = dst in _INT_DTYPES_
    is_src_uint = src in [d for d in _UINT_DTYPES_ if d is not None]
    is_dst_uint = dst in [d for d in _UINT_DTYPES_ if d is not None]
    is_src_float = src in _FLOAT_DTYPES_
    is_dst_float = dst in _FLOAT_DTYPES_

    # int <-> int
    if is_src_int and is_dst_int:
        return True
    # uint <-> uint
    if is_src_uint and is_dst_uint:
        return True
    # float <-> float
    if is_src_float and is_dst_float:
        return True
    # int <-> float
    if (is_src_int and is_dst_float):
        return True
    if (is_src_float and is_dst_int):
        return True    
    #  uint -> float
    if (is_src_uint and is_dst_float):
        return True
    #  uint -> int
    if (is_src_uint and is_dst_int):
        return True

    return False

class CastTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_tensor: torch.Tensor,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        output_dtype: torch.dtype,
        shape_output: List[int] | None,
        stride_output: List[int] | None,
    ):
        super().__init__("cast")
        self.input = input_tensor
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.output_dtype = output_dtype
        self.shape_output = shape_output
        self.stride_output = stride_output

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_output is not None:
            test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape_output)

        np_input = self.input.cpu().numpy()
        if np_input.dtype == np.uint32:
            stored_input = np_input.astype(np.int64)
        elif np_input.dtype == np.uint64:
            stored_input = np_input.astype(np.int64) 
        else:
            stored_input = np_input

        stored_input = np.ascontiguousarray(stored_input)
        contig_strides = contiguous_gguf_strides(self.shape_input)
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*contig_strides))

        rawid_in = np_dtype_to_ggml(stored_input.dtype)

        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            stored_input,
            raw_dtype=rawid_in,
        )

        try:
            output_torch = torch.zeros_like(self.input, dtype=self.output_dtype)
            np_output = output_torch.cpu().numpy()
        except Exception:
            np_map = {
                getattr(torch, "uint32", None): np.uint32,
                getattr(torch, "uint64", None): np.uint64,
            }
            np_dtype = np_map.get(self.output_dtype, None)
            if np_dtype is None:
                np_output = np.zeros(self.shape_output, dtype=np.int32)
            else:
                np_output = np.zeros(self.shape_output, dtype=np_dtype)

        if np_output.dtype == np.uint32:
            stored_out = np_output.astype(np.int64)
        elif np_output.dtype == np.uint64:
            stored_out = np_output.astype(np.int64)
        else:
            stored_out = np_output

        stored_out = np.ascontiguousarray(stored_out)
        contig_out_strides = contiguous_gguf_strides(self.shape_output)
        test_writer.add_array(test_writer.gguf_key("output.strides"), gguf_strides(*contig_out_strides))

        rawid_out = np_dtype_to_ggml(stored_out.dtype)

        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            stored_out,
            raw_dtype=rawid_out,
        )
        try:
            expected = self.input.to(self.output_dtype).double().cpu().numpy()
        except Exception:
            np_in = self.input.cpu().numpy()
            torch_to_np_map = {
                torch.int32: np.int32,
                torch.int64: np.int64,
                getattr(torch, "uint32", None): np.uint32,
                getattr(torch, "uint64", None): np.uint64,
                torch.float16: np.float16,
                torch.float32: np.float32,
                torch.float64: np.float64,
            }
            np_dst = torch_to_np_map.get(self.output_dtype, None)
            if np_dst is None:
                expected = np_in.astype(np.int32).astype(np.float64)
            else:
                expected = np_in.astype(np_dst).astype(np.float64)

        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            expected,
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )

def _dtype_suffix(dtype: torch.dtype) -> str:
    u32 = getattr(torch, "uint32", None)
    u64 = getattr(torch, "uint64", None)

    if u32 is not None and dtype == u32:
        return "u32"
    if u64 is not None and dtype == u64:
        return "u64"
    if dtype == torch.int32:
        return "i32"
    if dtype == torch.int64:
        return "i64"
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float16:
        return "f16"
    if getattr(torch, "bfloat16", None) is not None and dtype == torch.bfloat16:
        return "bf16"
    
    s = str(dtype)
    s = s.replace("torch.", "")
    s = s.replace("numpy.", "")
    s = s.replace("<class '", "").replace("'>", "")
    return s.replace(".", "_").replace(" ", "")

if __name__ == "__main__":
    _TEST_CASES_ = [
        # shape, input_stride, output_stride
        ((3, 3), None, None),
        ((32, 512), None, None),
        ((32, 512), (1024, 1), None),
        ((32, 512), (1024, 1), (1024, 1)),
        ((4, 4, 4), None, None),
        ((16, 32, 512), None, None),
        ((16, 20, 512), (20480, 512, 1), None),
        ((16, 20, 512), (20480, 512, 1), (20480, 512, 1)),
        ((1024,), None, None),
        ((1024,), (2,), None),
        ((1024,), (2,), (2,)),
        ((2, 3, 4, 5), None, None),
    ]

    for shape, stride_input, stride_output in _TEST_CASES_:
        for src_dtype in _ALL_DTYPES_:
            suffix = _dtype_suffix(src_dtype)
            filename = f"cast_{suffix}.gguf"
            test_writer = InfiniopTestWriter(filename)
            test_cases: List[CastTestCase] = []
            for dst_dtype in _ALL_DTYPES_:
                if not is_valid_cast(src_dtype, dst_dtype):
                    continue

                input_tensor = random_tensor(list(shape), src_dtype)
                output_dtype = dst_dtype

                test_case = CastTestCase(
                    input_tensor=input_tensor,
                    shape_input=list(shape),
                    stride_input=list(stride_input) if stride_input is not None else None,
                    output_dtype=output_dtype,
                    shape_output=list(shape),
                    stride_output=list(stride_output) if stride_output is not None else None,
                )
                test_cases.append(test_case)

            test_writer.add_tests(test_cases)
            test_writer.save()
