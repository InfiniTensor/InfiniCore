import ml_dtypes
import numpy as np
import torch

import infinicore


def to_torch_dtype(infini_dtype):
    """Convert infinicore data type to PyTorch data type"""
    if infini_dtype == infinicore.float16:
        return torch.float16
    elif infini_dtype == infinicore.float32:
        return torch.float32
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
    else:
        raise ValueError(f"Unsupported infinicore dtype: {infini_dtype}")


def to_infinicore_dtype(torch_dtype):
    """Convert PyTorch data type to infinicore data type"""
    if torch_dtype == torch.float32:
        return infinicore.float32
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
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def numpy_to_infinicore_dtype(numpy_dtype):
    """Convert numpy data type to infinicore data type"""
    if numpy_dtype == np.float32:
        return infinicore.float32
    elif numpy_dtype == np.float64:
        return infinicore.float64
    elif numpy_dtype == np.float16:
        return infinicore.float16
    elif numpy_dtype == ml_dtypes.bfloat16:
        return infinicore.bfloat16
    elif numpy_dtype == np.int8:
        return infinicore.int8
    elif numpy_dtype == np.int16:
        return infinicore.int16
    elif numpy_dtype == np.int32:
        return infinicore.int32
    elif numpy_dtype == np.int64:
        return infinicore.int64
    elif numpy_dtype == np.uint8:
        return infinicore.uint8
    else:
        raise ValueError(f"Unsupported numpy dtype: {numpy_dtype}")


def infinicore_to_numpy_dtype(infini_dtype):
    """Convert infinicore data type to numpy data type"""
    if infini_dtype == infinicore.float32:
        return np.float32
    elif infini_dtype == infinicore.float64:
        return np.float64
    elif infini_dtype == infinicore.float16:
        return np.float16
    elif infini_dtype == infinicore.int8:
        return np.int8
    elif infini_dtype == infinicore.int16:
        return np.int16
    elif infini_dtype == infinicore.bfloat16:
        return ml_dtypes.bfloat16
    elif infini_dtype == infinicore.int32:
        return np.int32
    elif infini_dtype == infinicore.int64:
        return np.int64
    elif infini_dtype == infinicore.uint8:
        return np.uint8
    else:
        raise ValueError(f"Unsupported infinicore dtype: {infini_dtype}")


def to_torch_device(infini_device):
    """Convert infinicore device type to PyTorch device type string
    
    Args:
        infini_device: infinicore.device object or device type string
        
    Returns:
        str: PyTorch device type string (e.g., "cpu", "cuda", "npu", "mlu", "musa")
    """
    from infinicore.device import device as infini_device_class
    from infinicore.lib import _infinicore
    
    # 如果输入是字符串，创建一个临时 device 对象
    if isinstance(infini_device, str):
        infini_device = infini_device_class(infini_device, 0)
    
    # 映射 infinicore device type 到 PyTorch device type
    _TORCH_DEVICE_MAP = {
        _infinicore.Device.Type.CPU: "cpu",
        _infinicore.Device.Type.NVIDIA: "cuda",
        _infinicore.Device.Type.CAMBRICON: "mlu",
        _infinicore.Device.Type.ASCEND: "npu",
        _infinicore.Device.Type.METAX: "cuda",
        _infinicore.Device.Type.MOORE: "musa",
        _infinicore.Device.Type.ILUVATAR: "cuda",
        _infinicore.Device.Type.KUNLUN: "cuda",
        _infinicore.Device.Type.HYGON: "cuda",
        _infinicore.Device.Type.QY: "cuda",
    }
    
    # 获取 underlying device type
    infinicore_device_type = infini_device._underlying.type
    
    # 转换为 PyTorch device type
    torch_device_type = _TORCH_DEVICE_MAP.get(infinicore_device_type)
    if torch_device_type is None:
        raise ValueError(f"Unsupported infinicore device type: {infinicore_device_type}")
    
    return torch_device_type
