import torch
import infinicore


def to_torch_dtype(infini_dtype):
    """Convert infinicore data type to PyTorch data type"""
    if infini_dtype == infinicore.float16:
        return torch.float16
    elif infini_dtype == infinicore.float32:
        return torch.float32
    elif infini_dtype == infinicore.float64:
        return torch.float64
    elif infini_dtype == infinicore.bfloat16:
        return torch.bfloat16
    elif infini_dtype == infinicore.int8:
        return torch.int8
    elif infini_dtype == infinicore.int16:
        return torch.int16
    elif infini_dtype == infinicore.int32:
        return torch.int32
    elif infini_dtype == infinicore.int64:
        return torch.int64
    elif infini_dtype == infinicore.uint8:
        return torch.uint8
    elif infini_dtype == infinicore.bool:
        return torch.bool
    elif infini_dtype == infinicore.complex64:
        return torch.complex64
    elif infini_dtype == infinicore.complex128:
        return torch.complex128
    else:
        raise ValueError(f"Unsupported infinicore dtype: {infini_dtype}")


def to_infinicore_dtype(torch_dtype):
    """Convert PyTorch data type to infinicore data type"""
    if torch_dtype == torch.float32:
        return infinicore.float32
    elif torch_dtype == torch.float64:
        return infinicore.float64
    elif torch_dtype == torch.float16:
        return infinicore.float16
    elif torch_dtype == torch.bfloat16:
        return infinicore.bfloat16
    elif torch_dtype == torch.int8:
        return infinicore.int8
    elif torch_dtype == torch.int16:
        return infinicore.int16
    elif torch_dtype == torch.int32:
        return infinicore.int32
    elif torch_dtype == torch.int64:
        return infinicore.int64
    elif torch_dtype == torch.uint8:
        return infinicore.uint8
    elif torch_dtype == torch.bool:
        return infinicore.bool
    elif torch_dtype == torch.complex64:
        return infinicore.complex64
    elif torch_dtype == torch.complex128:
        return infinicore.complex128
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
