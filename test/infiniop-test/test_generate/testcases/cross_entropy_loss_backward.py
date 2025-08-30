from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor
import torch

def cross_entropy_loss_backward(
    logits: np.ndarray,
    target: np.ndarray,
):
    torch_logits = torch.from_numpy(logits)
    torch_logits.requires_grad_(True)

    loss = torch.nn.CrossEntropyLoss()(
        torch_logits, 
        torch.from_numpy(target)
    )
    loss.backward()
    return torch_logits.grad.detach().numpy()


class CrossEntropyLossBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        logits: np.ndarray,
        probs: np.ndarray,
        shape_probs: List[int] | None,
        stride_probs: List[int] | None,
        target: np.ndarray,
        shape_target: List[int] | None,
        stride_target: List[int] | None,
        grad_logits: np.ndarray,
        shape_grad_logits: List[int] | None,
        stride_grad_logits: List[int] | None,

    ):
        super().__init__("cross_entropy_loss_backward")
        self.logits = logits
        self.probs = probs
        self.shape_probs = shape_probs
        self.stride_probs = stride_probs
        self.target = target
        self.shape_target = shape_target
        self.stride_target = stride_target
        self.grad_logits = grad_logits
        self.shape_grad_logits = shape_grad_logits
        self.stride_grad_logits = stride_grad_logits


    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("probs.shape"), self.shape_probs)
        test_writer.add_array(test_writer.gguf_key("target.shape"), self.shape_target)
        test_writer.add_array(test_writer.gguf_key("grad_logits.shape"), self.shape_grad_logits)
        if self.stride_probs is not None:
            test_writer.add_array(test_writer.gguf_key("probs.strides"), gguf_strides(*self.stride_probs))
        if self.stride_target is not None:
            test_writer.add_array(test_writer.gguf_key("target.strides"), gguf_strides(*self.stride_target))
        test_writer.add_array(
            test_writer.gguf_key("grad_logits.strides"),
            gguf_strides(*self.stride_grad_logits if self.stride_grad_logits is not None else contiguous_gguf_strides(self.shape_grad_logits))
        )
        test_writer.add_tensor(
            test_writer.gguf_key("probs"), self.probs, raw_dtype=np_dtype_to_ggml(self.probs.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("target"), self.target, raw_dtype=np_dtype_to_ggml(self.target.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_logits"), self.grad_logits, raw_dtype=np_dtype_to_ggml(self.grad_logits.dtype)
        )
        ans = cross_entropy_loss_backward(
            self.logits.astype(np.float64),
            self.target.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("cross_entropy_loss_backward.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((16, 5632), None, None, None),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    def get_batch_size(shape):
        batch_size = 1
        for d in range(len(shape) - 1):
            batch_size *= shape[d]  
        return batch_size      

    for dtype in _TENSOR_DTYPES_:
        for shape, stride_probs, stride_target, stride_grad_logits in _TEST_CASES_:
            logits = np.random.rand(*shape).astype(dtype) * 2 - 1
            probs = torch.softmax(torch.from_numpy(logits), dim=-1).detach().numpy()
            batch_size = get_batch_size(shape)
            target = torch.where(
                torch.arange(shape[-1]).unsqueeze(0).repeat(batch_size, 1) == \
                torch.randint(low=0, high=shape[-1], size=(batch_size,)).unsqueeze(1).repeat(1, shape[-1]),
                1, 0
            ).reshape(shape).detach().numpy().astype(dtype)    

            grad_logits = np.empty(tuple(0 for _ in shape), dtype=dtype)
            probs = process_zero_stride_tensor(probs, stride_probs)
            target = process_zero_stride_tensor(target, stride_target)
            test_case = CrossEntropyLossBackwardTestCase(
                logits=logits,
                probs=probs,
                shape_probs=shape,
                stride_probs=stride_probs,
                target=target,
                shape_target=shape,
                stride_target=stride_target,
                grad_logits=grad_logits,
                shape_grad_logits=shape,
                stride_grad_logits=stride_grad_logits,
            )
            test_cases.append(test_case)
            
    test_writer.add_tests(test_cases)
    test_writer.save()
    