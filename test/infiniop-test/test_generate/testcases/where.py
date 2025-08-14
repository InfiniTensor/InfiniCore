import torch
import gguf
import numpy as np
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

SUPPORTED_TORCH_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16
]

_TEST_CASES_ = [
    # shape, a_stride, b_stride, cond_stride, c_stride
    ((3, 3), None, None, None, None),
    ((32, 512), None, None, None, None),
    ((32, 512), (1024, 1), None, None, None),
    ((32, 512), (1024, 1), (1024, 1), None, None),
    ((32, 512), (1024, 1), (1024, 1), (1024, 1), None),
    ((32, 512), (1024, 1), (1024, 1), (1024, 1), (1024, 1)),
    ((4, 4, 4), None, None, None, None),
    ((16, 32, 512), None, None, None, None),
    ((16, 20, 512), (20480, 512, 1), None, None, None),
    ((16, 20, 512), (20480, 512, 1), (20480, 512, 1), None, None),
    ((16, 20, 512), (20480, 512, 1), (20480, 512, 1), (20480, 512, 1), None),
    ((16, 20, 512), (20480, 512, 1), (20480, 512, 1), (20480, 512, 1), (20480, 512, 1)),
    ((1024,), None, None, None, None),
    ((1024,), (2,), None, None, None),
    ((1024,), (2,), (2,), None, None),
    ((1024,), (2,), (2,), (2,), None),
    ((1024,), (2,), (2,), (2,), (2,)),
    ((2, 3, 4, 5), None, None, None, None),
]

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
    return (torch.rand(*shape, dtype=torch.float32) * 10) - 5

def tensor_to_gguf(t: torch.Tensor):
    if t.dtype == torch.bfloat16:
        bits = t.view(torch.uint16).cpu().numpy()
        arr = bits.view(bfloat16)
        raw = np_dtype_to_ggml(bfloat16)
        return arr, raw

    arr = t.cpu().numpy()
    dt = arr.dtype

    if dt == np.uint8:
        return arr.astype(np.int16), gguf.GGMLQuantizationType.I16
    if dt == np.uint32:
        return arr.astype(np.int64), gguf.GGMLQuantizationType.I64
    if dt == np.uint64:
        return arr.astype(np.float64), gguf.GGMLQuantizationType.F64

    raw = np_dtype_to_ggml(dt)
    return arr, raw

class WhereTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: torch.Tensor,
        shape_a: List[int] | None,
        stride_a: List[int] | None,
        b: torch.Tensor,
        shape_b: List[int] | None,
        stride_b: List[int] | None,
        condition: torch.Tensor,
        shape_condition: List[int] | None,
        stride_condition: List[int] | None,
        c: torch.Tensor,
        shape_c: List[int] | None,
        stride_c: List[int] | None,
    ):
        super().__init__("where")
        self.a = a
        self.shape_a = shape_a
        self.stride_a = stride_a

        self.b = b
        self.shape_b = shape_b
        self.stride_b = stride_b

        self.condition = condition
        self.shape_condition = shape_condition
        self.stride_condition = stride_condition

        self.c = c
        self.shape_c = shape_c
        self.stride_c = stride_c

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        if self.shape_a is not None:
            test_writer.add_array(test_writer.gguf_key("a.shape"), self.shape_a)
        if self.shape_b is not None:
            test_writer.add_array(test_writer.gguf_key("b.shape"), self.shape_b)
        if self.shape_condition is not None:
            test_writer.add_array(test_writer.gguf_key("condition.shape"), self.shape_condition)

        if self.shape_c is not None:
            test_writer.add_array(test_writer.gguf_key("c.shape"), self.shape_c)

        sa = self.stride_a if self.stride_a is not None else contiguous_gguf_strides(self.shape_a)
        sb = self.stride_b if self.stride_b is not None else contiguous_gguf_strides(self.shape_b)
        sc = self.stride_condition if self.stride_condition is not None else contiguous_gguf_strides(self.shape_condition)
        sc_out = self.stride_c if self.stride_c is not None else contiguous_gguf_strides(self.shape_c)

        test_writer.add_array(test_writer.gguf_key("a.strides"), gguf_strides(*sa))
        test_writer.add_array(test_writer.gguf_key("b.strides"), gguf_strides(*sb))
        test_writer.add_array(test_writer.gguf_key("condition.strides"), gguf_strides(*sc))
        test_writer.add_array(test_writer.gguf_key("c.strides"), gguf_strides(*sc_out))

        a_arr, a_raw = tensor_to_gguf(self.a)
        test_writer.add_tensor(test_writer.gguf_key("a"), a_arr, raw_dtype=a_raw)

        b_arr, b_raw = tensor_to_gguf(self.b)
        test_writer.add_tensor(test_writer.gguf_key("b"), b_arr, raw_dtype=b_raw)

        a_torch_dt = self.a.dtype
        target_cond_torch_dt = a_torch_dt
        if target_cond_torch_dt == torch.bfloat16:
            cond_same = self.condition.to(dtype=torch.float32).to(dtype=torch.bfloat16)
        else:
            cond_same = self.condition.to(dtype=target_cond_torch_dt)
        cond_arr, cond_raw = tensor_to_gguf(cond_same)

        test_writer.add_tensor(test_writer.gguf_key("condition"), cond_arr, raw_dtype=cond_raw)

        c_arr, c_raw = tensor_to_gguf(self.c)
        test_writer.add_tensor(test_writer.gguf_key("c"), c_arr, raw_dtype=c_raw)

        a_ref = self.a if isinstance(self.a, torch.Tensor) else torch.tensor(self.a, dtype=torch.float64)
        b_ref = self.b if isinstance(self.b, torch.Tensor) else torch.tensor(self.b, dtype=torch.float64)
        ans = torch.where(self.condition.to(torch.bool), a_ref, b_ref).double()
        test_writer.add_tensor(test_writer.gguf_key("ans"), ans.numpy(), raw_dtype=gguf.GGMLQuantizationType.F64)

def _dtype_suffix(dtype: torch.dtype) -> str:
    if dtype == torch.int8:
        return "i8"
    if dtype == torch.int16:
        return "i16"
    if dtype == torch.int32:
        return "i32"
    if dtype == torch.int64:
        return "i64"
    if dtype == torch.float64:
        return "f64"
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    s = str(dtype)
    s = s.replace("torch.", "")
    s = s.replace("numpy.", "")
    s = s.replace("<class '", "").replace("'>", "")
    return s.replace(".", "_").replace(" ", "")

if __name__ == "__main__":
    for dtype in SUPPORTED_TORCH_DTYPES:
        suffix = _dtype_suffix(dtype)
        filename = f"where_{suffix}.gguf"
        test_writer = InfiniopTestWriter(filename)
        test_cases: List[WhereTestCase] = []
        for shape, stride_a, stride_b, stride_cond, stride_c in _TEST_CASES_:
            a_tensor = random_tensor(list(shape), dtype)
            b_tensor = random_tensor(list(shape), dtype)
            cond_bool = random_tensor(list(shape), dtype=torch.bool)
            c_tensor = torch.empty_like(a_tensor)

            test_case = WhereTestCase(
                a=a_tensor,
                shape_a=list(shape),
                stride_a=list(stride_a) if stride_a else None,
                b=b_tensor,
                shape_b=list(shape),
                stride_b=list(stride_b) if stride_b else None,
                condition=cond_bool,
                shape_condition=list(shape),
                stride_condition=list(stride_cond) if stride_cond else None,
                c=c_tensor,
                shape_c=list(shape),
                stride_c=list(stride_c) if stride_c else None,
            )
            test_cases.append(test_case)

        test_writer.add_tests(test_cases)
        test_writer.save()
