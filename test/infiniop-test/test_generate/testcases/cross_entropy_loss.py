import numpy as np
import gguf
import torch
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def cross_entropy_loss(
    input_logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    """Cross Entropy Loss using PyTorch with double precision"""
    return F.cross_entropy(
        input_logits.double(),
        target,
        weight=weight,
        size_average=size_average,
        ignore_index=ignore_index,
        reduce=reduce,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


def random_tensor(shape, dtype):
    """Generate random tensor"""
    if dtype == torch.bfloat16:
        tensor = torch.randn(shape, dtype=torch.float32).to(dtype).clamp(-2, 2)
    else:
        tensor = torch.randn(shape, dtype=dtype).clamp(-2, 2)
    return tensor


def random_target(shape, num_classes):
    """Generate random target tensor with class indices"""
    return torch.randint(0, num_classes, shape, dtype=torch.long)


class CrossEntropyLossTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__("cross_entropy_loss")

        # Generate logits tensor
        if len(input_shape) == 1:
            # Shape (C,) - single sample classification
            logits_shape = (num_classes,)
            self.target_shape = ()
        elif len(input_shape) >= 2:
            # Shape (N, C, [d1], [d2], ...)
            if input_shape[1] != num_classes:
                raise ValueError(
                    f"Input shape {input_shape} does not match num_classes {num_classes}."
                )
            logits_shape = input_shape
            self.target_shape = (input_shape[0],) + input_shape[2:]
        else:
            raise ValueError(
                f"Unsupported input shape: {input_shape}."
            )

        self.input_logits = random_tensor(logits_shape, dtype)
        self.target = random_target(self.target_shape, num_classes)

        # Compute loss using default parameters
        self.loss = cross_entropy_loss(
            self.input_logits,
            self.target,
            weight=None,
            size_average=None,
            ignore_index=-100,
            reduce=None,
            reduction="mean",
            label_smoothing=0.0,
        )

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # Helper function to handle data type conversion for logits
        def convert_tensor(tensor):
            if tensor.dtype == torch.bfloat16:
                return (
                    tensor.view(torch.uint16).detach().numpy(),
                    gguf.GGMLQuantizationType.BF16,
                )
            else:
                return tensor.detach().numpy(), np_dtype_to_ggml(
                    tensor.detach().numpy().dtype
                )

        # Add logits tensor - keep original data type
        logits_numpy, ggml_dtype_logits = convert_tensor(self.input_logits)
        test_writer.add_tensor(
            test_writer.gguf_key("logits"),
            logits_numpy,
            raw_dtype=ggml_dtype_logits,
        )
        test_writer.add_array(
            test_writer.gguf_key("logits_shape"), 
            list(self.input_logits.shape)
        )

        # Add target tensor - always int64
        test_writer.add_tensor(
            test_writer.gguf_key("target"),
            self.target.detach().numpy().astype(np.int64),
            raw_dtype=gguf.GGMLQuantizationType.I64,
        )
        test_writer.add_array(
            test_writer.gguf_key("target_shape"), 
            list(self.target.shape)
        )

        # Add loss output - use double precision
        test_writer.add_tensor(
            test_writer.gguf_key("loss"),
            self.loss.detach().numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )
        test_writer.add_array(
            test_writer.gguf_key("loss_shape"), 
            list(self.loss.shape)
        )


def gen_gguf(dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    # Format: (input_shape, num_classes)
    _TEST_CASES = [
        # Single sample classification
        ((10,), 10),
        ((1000,), 200),
        
        # 2D: (N, C) - batch classification
        ((4, 10), 10),
        ((8, 5), 5),
        ((16, 100), 100),
        ((32, 1000), 1000),
        ((64, 21), 21),
        ((128, 50), 50),
        
        # 3D: (N, C, d1) - sequence classification
        ((4, 10, 5), 10),
        
        # 4D: (N, C, d1, d2) - image segmentation
        ((2, 8, 8, 8), 8),
        
        # 5D: (N, C, d1, d2, d3) - 3D segmentation
        ((3, 10, 10, 20, 30), 10),
    ]

    for input_shape, num_classes in _TEST_CASES:
        test_case = CrossEntropyLossTestCase(
            input_shape=input_shape,
            num_classes=num_classes,
            dtype=dtype,
        )
        test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()


if __name__ == "__main__":
    # Data types to test
    _TENSOR_DTYPES_ = [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]
    
    dtype_filename_map = {
        torch.float32: "cross_entropy_loss_float32.gguf",
        torch.float16: "cross_entropy_loss_float16.gguf",
        torch.bfloat16: "cross_entropy_loss_bfloat16.gguf",
    }

    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        print(f"Generating {filename} for dtype {dtype}")
        gen_gguf(dtype, filename)
        
    print(f"Generated GGUF files for {len(_TENSOR_DTYPES_)} data types")
    print("Test cases include single sample, batch, sequence, and multi-dimensional classification. All tensors use contiguous memory.")
