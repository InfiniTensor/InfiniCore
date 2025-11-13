import ctypes
import numbers
from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref, addressof

import infinicore

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional
    np = None

from infinicore.device import device as Device
from infinicore.dtype import (
    bool as bool_dtype,
    complex128,
    complex32,
    complex64,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
from infinicore.dtype import dtype
from infinicore.lib import _infinicore

from .utils import to_infinicore_dtype

_DTYPE_ITEMSIZE = {
    bool_dtype: ctypes.sizeof(ctypes.c_bool),
    int8: ctypes.sizeof(ctypes.c_int8),
    int16: ctypes.sizeof(ctypes.c_int16),
    int32: ctypes.sizeof(ctypes.c_int32),
    int64: ctypes.sizeof(ctypes.c_int64),
    uint8: ctypes.sizeof(ctypes.c_uint8),
    uint16: ctypes.sizeof(ctypes.c_uint16),
    uint32: ctypes.sizeof(ctypes.c_uint32),
    uint64: ctypes.sizeof(ctypes.c_uint64),
    float16: 2,
    float32: ctypes.sizeof(ctypes.c_float),
    float64: ctypes.sizeof(ctypes.c_double),
    complex32: 4,
    complex64: 8,
    complex128: 16,
}


class Tensor:
    def __init__(self, underlying, *, _torch_ref=None):
        """An internal method. Please do not use this directly."""

        self._underlying = underlying

        from infinicore.dtype import dtype
        self._dtype = dtype(self._underlying.dtype)

        self._device = Device._from_infinicore_device(
            self._underlying.device
        )

        self._torch_ref = _torch_ref

    @property
    def shape(self):
        return self._underlying.shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def ndim(self):
        return self._underlying.ndim

    def data_ptr(self):
        return self._underlying.data_ptr()

    def size(self, dim=None):
        if dim is None:
            return self.shape

        return self.shape[dim]

    def stride(self, dim=None):
        if dim is None:
            return self._underlying.strides

        return self._underlying.strides[dim]

    def numel(self):
        return self._underlying.numel()

    def is_contiguous(self):
        return self._underlying.is_contiguous()

    def is_is_pinned(self):
        return self._underlying.is_is_pinned()

    def copy_(self, src):
        self._underlying.copy_(src._underlying)

    def to(self, *args, **kwargs):
        return Tensor(
            self._underlying.to(*tuple(arg._underlying for arg in args), **kwargs)
        )

    def as_strided(self, size, stride):
        return Tensor(self._underlying.as_strided(size, stride))

    def contiguous(self):
        return Tensor(self._underlying.contiguous())

    def permute(self, dims):
        return Tensor(self._underlying.permute(dims))

    def view(self, shape):
        return Tensor(self._underlying.view(shape))

    def debug(self, filename=None):
        """Print tensor data or save to file for debugging

        Args:
            filename: Optional filename to save raw binary data. If None, prints to stdout.
        """
        if filename is None:
            self._underlying.debug()
        else:
            self._underlying.debug(filename)

    def __add__(self, other):
        return infinicore.add(self, other)

    def __mul__(self, other):
        return infinicore.mul(self, other)


def empty(size, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.empty(size, dtype._underlying, device._underlying, pin_memory)
    )


def empty_like(input, *, dtype=None, device=None):
    if dtype is None:
        dtype = input.dtype

    if device is None:
        device = input.device

    return empty(input.size(), dtype=dtype, device=device)


def strided_empty(size, strides, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.strided_empty(
            size, strides, dtype._underlying, device._underlying, pin_memory
        )
    )


def zeros(size, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.zeros(size, dtype._underlying, device._underlying, pin_memory)
    )


def ones(size, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.ones(size, dtype._underlying, device._underlying, pin_memory)
    )


def from_blob(data_ptr, size, *, dtype=None, device=None):
    return Tensor(
        _infinicore.from_blob(data_ptr, size, dtype._underlying, device._underlying)
    )


def strided_from_blob(data_ptr, size, strides, *, dtype=None, device=None):
    return Tensor(
        _infinicore.strided_from_blob(
            data_ptr, size, strides, dtype._underlying, device._underlying
        )
    )


def from_torch(torch_tensor) -> Tensor:
    infini_type = to_infinicore_dtype(torch_tensor.dtype)
    infini_device = infinicore.device(torch_tensor.device.type, 0)
    return Tensor(
        _infinicore.from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            dtype=infini_type._underlying,
            device=infini_device._underlying,
        ),
        _torch_ref=torch_tensor,
    )


def _py_to_ctype(py_type):
    """将Python类型映射到ctypes类型"""
    type_map = {
        bool: ctypes.c_bool,
        int: ctypes.c_int64,
        float: ctypes.c_double,
        complex: ctypes.c_double * 2,  # 复数用两个double表示
    }
    
    # 处理numpy类型（如果可用）
    try:
        import numpy as np
        type_map.update({
            np.bool_: ctypes.c_bool,
            np.int8: ctypes.c_int8,
            np.int16: ctypes.c_int16,
            np.int32: ctypes.c_int32,
            np.int64: ctypes.c_int64,
            np.uint8: ctypes.c_uint8,
            np.uint16: ctypes.c_uint16,
            np.uint32: ctypes.c_uint32,
            np.uint64: ctypes.c_uint64,
            np.float16: ctypes.c_uint16,  # float16需要特殊处理
            np.float32: ctypes.c_float,
            np.float64: ctypes.c_double,
            np.complex64: ctypes.c_float * 2,
            np.complex128: ctypes.c_double * 2,
        })
    except ImportError:
        pass
    
    return type_map.get(py_type, ctypes.c_double)


def _normalize_scalar(value):
    """将可能的 numpy scalar 转换为原生 Python 类型。"""
    if value is None:
        return None
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
    return value


def _require_numpy():
    if np is None:
        raise ImportError(
            "此数据类型的转换依赖 numpy，请先安装 numpy 后再试。"
        )
    return np


def _infer_value_category(data):
    """遍历全部数据，判断需要的数值类型类别。"""
    flags = {"complex": False, "float": False, "int": False, "bool": False}

    def _scan(lst):
        for item in lst:
            if isinstance(item, list):
                _scan(item)
                continue

            val = _normalize_scalar(item)
            if val is None:
                continue

            if isinstance(val, bool):
                flags["bool"] = True
                continue

            if isinstance(val, complex):
                flags["complex"] = True
                continue

            if isinstance(val, numbers.Real):
                # bool 已经处理，这里只剩下纯整型或浮点
                if isinstance(val, numbers.Integral):
                    flags["int"] = True
                else:
                    flags["float"] = True
                continue

            # 其它类型，视为浮点
            flags["float"] = True

    _scan(data)

    if flags["complex"]:
        return "complex"
    if flags["float"]:
        return "float"
    if flags["int"]:
        return "int"
    if flags["bool"]:
        return "bool"
    return None


def _convert_scalar_for_dtype(val, dtype):
    if dtype in (float32, float64):
        return float(val)
    if dtype in (int8, int16, int32, int64, uint8, uint16, uint32, uint64):
        return int(val)
    if dtype == bool_dtype:
        return bool(val)
    return val


def _infer_shape_and_dtype(data, shape=None, dtype=None):
    """推断多维list的shape和dtype"""
    def _get_shape(lst, current_shape=None):
        """递归获取shape"""
        if current_shape is None:
            current_shape = []
        
        if not isinstance(lst, list) or len(lst) == 0:
            return current_shape
        
        current_shape.append(len(lst))
        
        # 检查第一个元素是否是list
        if isinstance(lst[0], list):
            return _get_shape(lst[0], current_shape)
        else:
            return current_shape
    
    def _validate_shape(lst, shape, dim=0):
        """验证所有维度的长度是否一致"""
        if dim >= len(shape):
            return True
        
        if len(lst) != shape[dim]:
            return False
        
        if dim < len(shape) - 1:
            for sublist in lst:
                if not isinstance(sublist, list):
                    return False
                if not _validate_shape(sublist, shape, dim + 1):
                    return False
        
        return True
    
    # 推断shape
    inferred_shape = _get_shape(data) if shape is None else shape
    
    # 推断dtype（如果未提供）
    if dtype is None:
        category = _infer_value_category(data)
        if category == "bool":
            inferred_dtype = bool_dtype
        elif category == "complex":
            inferred_dtype = complex64
        else:
            # 默认与 torch.Tensor 保持一致：整数/浮点都使用 float32
            inferred_dtype = float32
    else:
        # 如果提供了dtype，直接使用
        inferred_dtype = dtype
    
    # 验证shape
    if not _validate_shape(data, inferred_shape):
        raise ValueError(f"不一致的shape: 期望 {inferred_shape}, 但数据不符合")
    
    return inferred_shape, inferred_dtype


def _flatten_list(data, shape, dtype):
    """将多维list展平为一维数组，并转换为ctypes数组"""
    def _flatten_recursive(lst, result):
        """递归展平list"""
        for item in lst:
            if isinstance(item, list):
                _flatten_recursive(item, result)
            else:
                result.append(item)
    
    flattened = []
    _flatten_recursive(data, flattened)
    
    if len(flattened) == 0:
        raise ValueError("展平后的数据为空")
    
    # 根据dtype确定ctypes类型
    # 从infinicore dtype映射回Python类型，然后映射到ctypes
    dtype_to_ctype_map = {
        bool_dtype: ctypes.c_bool,
        int8: ctypes.c_int8,
        int16: ctypes.c_int16,
        int32: ctypes.c_int32,
        int64: ctypes.c_int64,
        uint8: ctypes.c_uint8,
        uint16: ctypes.c_uint16,
        uint32: ctypes.c_uint32,
        uint64: ctypes.c_uint64,
        float32: ctypes.c_float,
        float64: ctypes.c_double,
    }

    # 特殊类型处理
    if dtype == float16:
        np_mod = _require_numpy()
        np_array = np_mod.asarray(flattened, dtype=np_mod.float16)
        uint_values = np_array.view(np_mod.uint16).tolist()
        storage = (ctypes.c_uint16 * len(uint_values))(*uint_values)
        return storage, addressof(storage), len(flattened)

    if dtype == complex32:
        np_mod = _require_numpy()
        np_array = np_mod.asarray(flattened, dtype=np_mod.complex64)
        real = np_array.real.astype(np_mod.float16)
        imag = np_array.imag.astype(np_mod.float16)
        packed = np_mod.stack([real, imag], axis=1).reshape(-1)
        uint_values = packed.view(np_mod.uint16).tolist()
        storage = (ctypes.c_uint16 * len(uint_values))(*uint_values)
        return storage, addressof(storage), len(flattened)

    if dtype == complex64:
        np_mod = _require_numpy()
        np_array = np_mod.asarray(flattened, dtype=np_mod.complex64)
        storage = (ctypes.c_float * (len(np_array) * 2))()
        for i, val in enumerate(np_array):
            storage[2 * i] = float(val.real)
            storage[2 * i + 1] = float(val.imag)
        return storage, addressof(storage), len(flattened)

    if dtype == complex128:
        np_mod = _require_numpy()
        np_array = np_mod.asarray(flattened, dtype=np_mod.complex128)
        storage = (ctypes.c_double * (len(np_array) * 2))()
        for i, val in enumerate(np_array):
            storage[2 * i] = float(val.real)
            storage[2 * i + 1] = float(val.imag)
        return storage, addressof(storage), len(flattened)

    # 常规类型
    ctype = dtype_to_ctype_map.get(dtype)
    if ctype is None:
        py_type = type(flattened[0])
        ctype = _py_to_ctype(py_type)

    try:
        array = (ctype * len(flattened))(*flattened)
    except (TypeError, OverflowError):
        converted = [_convert_scalar_for_dtype(val, dtype) for val in flattened]
        array = (ctype * len(converted))(*converted)

    return array, addressof(array), len(flattened)


def _create_cpu_tensor_from_list(shape, infini_dtype, data):
    cpu_device = Device("cpu", 0)

    ctypes_array, data_ptr, value_count = _flatten_list(data, shape, infini_dtype)

    tensor_cpu = empty(
        shape,
        dtype=infini_dtype,
        device=cpu_device,
        pin_memory=False,
    )

    itemsize = _DTYPE_ITEMSIZE.get(infini_dtype)
    if itemsize is None:
        raise ValueError(f"暂不支持 dtype {infini_dtype} 的 memmove 拷贝")

    total_bytes = value_count * itemsize
    if total_bytes:
        ctypes.memmove(tensor_cpu.data_ptr(), data_ptr, total_bytes)

    return tensor_cpu


def tensor(data, *, dtype=None, device=None):
    """
    从Python list创建infinicore.Tensor，类似于torch.Tensor的功能。
    
    Args:
        data: Python list或嵌套list，支持多维数组
        dtype: 可选的infinicore dtype，如果不提供则自动推断
        device: 可选的infinicore device，默认为CPU
    
    Returns:
        infinicore.Tensor: 创建的张量
    
    Examples:
        >>> import infinicore
        >>> t1 = infinicore.tensor([1, 2, 3])
        >>> t2 = infinicore.tensor([[1, 2], [3, 4]])
        >>> t3 = infinicore.tensor([1.0, 2.0, 3.0], dtype=infinicore.float32)
    """
    if not isinstance(data, list):
        raise TypeError(f"期望list类型，但得到 {type(data)}")
    
    if len(data) == 0:
        raise ValueError("不能从空list创建tensor")
    
    # 设置默认device
    if device is None:
        device = Device("cpu", 0)
    elif isinstance(device, str):
        device = Device(device)
    
    shape, infini_dtype = _infer_shape_and_dtype(data, dtype=dtype)

    if device.type != "cpu":
        raise NotImplementedError(
            "python.list -> infinicore.Tensor 当前仅支持 CPU 设备，请先创建 CPU Tensor 后调用 .to(device)"
        )

    return _create_cpu_tensor_from_list(shape, infini_dtype, data)
