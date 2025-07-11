import torch
import gguf
import numpy as np
from typing import List, Union, Tuple

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

_INT_DTYPES_   = [torch.int32, torch.int64]
_UINT_DTYPES_  = [torch.uint8, torch.uint32, torch.uint64]
_FLOAT_DTYPES_ = [torch.float16, torch.float32, torch.float64]
_BFLOAT_DTYPES_= [torch.bfloat16]
_ALL_DTYPES_   = _INT_DTYPES_ + _UINT_DTYPES_ + _FLOAT_DTYPES_ + _BFLOAT_DTYPES_

def is_broadcastable(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> bool:
    len1, len2 = len(shape1), len(shape2)
    for i in range(1, max(len1, len2) + 1):
        d1 = shape1[-i] if i <= len1 else 1
        d2 = shape2[-i] if i <= len2 else 1
        if not (d1 == d2 or d1 == 1 or d2 == 1):
            return False
    return True

def reference_where(condition, a, b):
    return torch.where(condition, a, b)

# np_dtype_to_ggml not supported for now
def adapt_for_gguf(t: torch.Tensor) -> Tuple[np.ndarray, gguf.GGMLQuantizationType]:
    if t.dtype == torch.bfloat16:
        np_arr = t.view(torch.uint16).numpy()
        return np_arr, gguf.GGMLQuantizationType.BF16
    np_arr = t.numpy()
    if np_arr.dtype == np.uint8:
        return np_arr.astype(np.int16), gguf.GGMLQuantizationType.I16
    if np_arr.dtype == np.uint32:
        return np_arr.astype(np.int64), gguf.GGMLQuantizationType.I64
    if np_arr.dtype == np.uint64:
        return np_arr.astype(np.float64), gguf.GGMLQuantizationType.F64

    return np_arr, np_dtype_to_ggml(np_arr.dtype)

class WhereTestCase(InfiniopTestCase):
    def __init__(
        self,
        condition: torch.Tensor,
        shape_condition: List[int],
        stride_condition: List[int] | None,
        a: Union[torch.Tensor, float],
        shape_a: List[int] | None,
        stride_a: List[int] | None,
        b: Union[torch.Tensor, float],
        shape_b: List[int] | None,
        stride_b: List[int] | None,
    ):
        super().__init__("where")
        self.condition, self.shape_condition, self.stride_condition = condition, shape_condition, stride_condition
        self.a, self.shape_a, self.stride_a = a, shape_a, stride_a
        self.b, self.shape_b, self.stride_b = b, shape_b, stride_b

    def write_test(self, test_writer: InfiniopTestWriter):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("condition.shape"),  self.shape_condition)
        sc = self.stride_condition or contiguous_gguf_strides(self.shape_condition)
        test_writer.add_array(test_writer.gguf_key("condition.strides"), gguf_strides(*sc))
        cond_np, cond_dtype = adapt_for_gguf(self.condition.to(torch.uint8))
        test_writer.add_tensor(test_writer.gguf_key("condition"), cond_np, raw_dtype=cond_dtype)

        if isinstance(self.a, torch.Tensor):
            test_writer.add_array(test_writer.gguf_key("a.shape"),  self.shape_a)            
            sa = self.stride_a or contiguous_gguf_strides(self.shape_a)
            test_writer.add_array(test_writer.gguf_key("a.strides"), gguf_strides(*sa))
            a_np, a_dtype = adapt_for_gguf(self.a)
            test_writer.add_tensor(test_writer.gguf_key("a"), a_np, raw_dtype=a_dtype)
        else:
            test_writer.add_float64(test_writer.gguf_key("a"), float(self.a))

        if isinstance(self.b, torch.Tensor):
            test_writer.add_array(test_writer.gguf_key("b.shape"),  self.shape_b)            
            sb = self.stride_b or contiguous_gguf_strides(self.shape_b)
            test_writer.add_array(test_writer.gguf_key("b.strides"), gguf_strides(*sb))
            b_np, b_dtype = adapt_for_gguf(self.b)
            test_writer.add_tensor(test_writer.gguf_key("b"), b_np, raw_dtype=b_dtype)
        else:
            test_writer.add_float64(test_writer.gguf_key("b"), float(self.b))

        c = reference_where(self.condition, self.a, self.b).double()
        test_writer.add_tensor(
            test_writer.gguf_key("c"),
            c.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )

def random_tensor(shape: List[int], dtype: torch.dtype) -> torch.Tensor:
    if dtype in _INT_DTYPES_:
        return torch.randint(-100, 100, shape, dtype=dtype)
    if dtype == torch.uint8:
        return torch.randint(0, 200, shape, dtype=dtype)
    if dtype in [torch.uint32, torch.uint64]:
        npmap = {torch.uint32: np.uint32, torch.uint64: np.uint64}
        return torch.from_numpy(np.random.randint(0, 200, size=shape, dtype=npmap[dtype]))
    if dtype in _FLOAT_DTYPES_ + _BFLOAT_DTYPES_:
        return torch.rand(*shape, dtype=dtype) * 10 - 5

if __name__ == "__main__":
    writer = InfiniopTestWriter("where.gguf")
    test_case: List[WhereTestCase] = []

    _TEST_CASES_ = [
        ((1,), None),
        ((4,), (2,)),
        ((4,4), None),
        ((4,4), (4,1)),
        ((2,3,4), None),
        ((2,3,4), (48,16,1)),
        ((2,1,4,1), None),
        ((3,2,5,4), (40,20,4,1)),
    ]

    for shape_condition, stride_condition in _TEST_CASES_:
        cond = torch.randint(0,2, shape_condition, dtype=torch.bool)

        for dtype in _ALL_DTYPES_:
            # not implemented for 'UInt32'
            if dtype in (torch.uint32, torch.uint64):
                continue

            for shape_a, stride_a in _TEST_CASES_:
                if not is_broadcastable(shape_condition, shape_a):
                    continue

                for shape_b, stride_b in _TEST_CASES_:
                    if not is_broadcastable(shape_condition, shape_b):
                        continue

                    if not is_broadcastable(shape_a, shape_b):
                        continue

                    a_t = random_tensor(shape_a, dtype)
                    b_t = random_tensor(shape_b, dtype)
                    test_case.append(WhereTestCase(
                        condition=cond,
                        shape_condition=list(shape_condition), stride_condition=list(stride_condition) if stride_condition else None,
                        a=a_t,        shape_a=list(shape_a), stride_a=list(stride_a) if stride_a else None,
                        b=b_t,        shape_b=list(shape_b), stride_b=list(stride_b) if stride_b else None,
                    ))

                    test_case.append(WhereTestCase(
                        condition=cond,
                        shape_condition=list(shape_condition), stride_condition=list(stride_condition) if stride_condition else None,
                        a=3.14,      shape_a=None,          stride_a=None,
                        b=b_t,       shape_b=list(shape_b), stride_b=list(stride_b) if stride_b else None,
                    ))

                    test_case.append(WhereTestCase(
                        condition=cond,
                        shape_condition=list(shape_condition), stride_condition=list(stride_condition) if stride_condition else None,
                        a=a_t,       shape_a=list(shape_a), stride_a=list(stride_a) if stride_a else None,
                        b=-2.718,    shape_b=None,          stride_b=None,
                    ))

    writer.add_tests(test_case)
    writer.save()