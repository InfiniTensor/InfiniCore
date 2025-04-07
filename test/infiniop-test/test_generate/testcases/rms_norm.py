from ast import List
import numpy as np
import gguf
from typing import Optional
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides

def create_non_contiguous(shape, dtype, stride_scale=2):
    expanded_shape = (shape[0] * stride_scale,) + shape[1:]
    buffer = np.random.uniform(-1.0, 1.0, expanded_shape).astype(dtype) * 0.001

    new_strides = (buffer.strides[0] * stride_scale,) + buffer.strides[1:]
    
    return as_strided(buffer, shape=shape, strides=new_strides)

def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, shape).astype(dtype) * 0.001

def rms_norm(input: np.ndarray, weight: np.ndarray, epsilon: float) -> np.ndarray:
    """
    使用numpy计算rms_norm结果
    Args:
        input:  输入张量, 维度为2, 形状为 [..., hidden_size]
        weight: 缩放权重, 形状为 [hidden_size]
        epsilon: 避免除零的小常数
    Returns:
        输出张量, 形状与 input 相同
    """
    squared = input ** 2
    mean = np.mean(squared, axis=-1, keepdims=True)
    rms = np.sqrt(mean + epsilon)
    
    normalized = input / rms
    return normalized * weight

class RMSNormTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_shape: tuple,
        weight_shape: tuple,
        atype: np.dtype,
        wtype: np.dtype,
        epsilon: float = 1e-5,
        input_non_contiguous: bool = False,
        input_stride_scale: int = 2,
    ):
        super().__init__("rms_norm")
        if input_non_contiguous:
            self.input = create_non_contiguous(input_shape, atype, input_stride_scale)
        else:
            self.input = random_tensor(input_shape, atype)
        self.weight = random_tensor(weight_shape, wtype)
        self.epsilon = epsilon
        self.result = np.zeros_like(self.input)
        self.ans = rms_norm(self.input, self.weight, self.epsilon).astype(atype)

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_float32(test_writer.gguf_key("epsilon"), self.epsilon)
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            self.input,
            raw_dtype=np_dtype_to_ggml(self.input.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("weight"),
            self.weight,
            raw_dtype=np_dtype_to_ggml(self.weight.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            self.ans,
            raw_dtype=np_dtype_to_ggml(self.ans.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("result"),
            self.result,
            raw_dtype=np_dtype_to_ggml(self.result.dtype),
        )

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("rms_norm.gguf")
    
    test_cases = [
        RMSNormTestCase(
            input_shape=(2, 256), 
            weight_shape=(256,),
            atype=np.float32,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            input_shape=(4, 512),
            weight_shape=(512,),
            atype=np.float32,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            input_shape=(8, 1024),
            weight_shape=(1024,),
            atype=np.float32,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            input_shape=(1, 768),
            weight_shape=(768,),
            atype=np.float32,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            input_shape=(8, 256), 
            weight_shape=(256,),
            atype=np.float32,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            input_shape=(500, 4096), 
            weight_shape=(4096,),
            atype=np.float32,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            input_shape=(2, 256),
            weight_shape=(256,),
            atype=np.float16,
            wtype=np.float16,
        ),
        RMSNormTestCase(
            input_shape=(4, 512),
            weight_shape=(512,),
            atype=np.float16,
            wtype=np.float16,
        ),
        RMSNormTestCase(
            input_shape=(500, 4096), 
            weight_shape=(4096,),
            atype=np.float16,
            wtype=np.float16,
        ),
        RMSNormTestCase(
            input_shape=(4, 512),
            weight_shape=(512,),
            atype=np.float16,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            input_shape=(500, 4096), 
            weight_shape=(4096,),
            atype=np.float16,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            input_shape=(4, 512),
            weight_shape=(512,),
            atype=np.float32,
            wtype=np.float32,
            input_non_contiguous=True,
            input_stride_scale=2,
        ),
        RMSNormTestCase(
            input_shape=(500, 4096),
            weight_shape=(4096,),
            atype=np.float32,
            wtype=np.float32,
            input_non_contiguous=True,
            input_stride_scale=2,
        ),
        RMSNormTestCase(
            input_shape=(4, 512),
            weight_shape=(512,),
            atype=np.float16,
            wtype=np.float16,
            input_non_contiguous=True,
            input_stride_scale=2,
        ),
        RMSNormTestCase(
            input_shape=(500, 4096),
            weight_shape=(4096,),
            atype=np.float16,
            wtype=np.float16,
            input_non_contiguous=True,
            input_stride_scale=2,
        ),
        RMSNormTestCase(
            input_shape=(4, 512),
            weight_shape=(512,),
            atype=np.float16,
            wtype=np.float32,
            input_non_contiguous=True,
            input_stride_scale=2,
        ),
        RMSNormTestCase(
            input_shape=(500, 4096),
            weight_shape=(4096,),
            atype=np.float16,
            wtype=np.float32,
            input_non_contiguous=True,
            input_stride_scale=2,
        ),
    ]
    
    test_writer.add_tests(test_cases)
    test_writer.save()
