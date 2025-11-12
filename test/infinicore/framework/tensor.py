import torch
import math
from pathlib import Path
from .datatypes import to_torch_dtype
from .devices import torch_device_map
from .utils import is_integer_dtype, is_complex_dtype


class TensorInitializer:
    """Tensor data initializer with multiple modes"""

    RANDOM = "random"
    ZEROS = "zeros"
    ONES = "ones"
    RANDINT = "randint"
    MANUAL = "manual"
    BINARY = "binary"
    FROM_FILE = "from_file"

    @staticmethod
    def create_tensor(shape, dtype, device, mode=RANDOM, strides=None, **kwargs):
        """
        Unified tensor creation interface for both contiguous and non-contiguous tensors

        Args:
            shape: Tensor shape
            dtype: infinicore dtype
            device: InfiniDeviceEnum
            mode: Initialization mode
            strides: Optional strides for non-contiguous tensors
            **kwargs: Additional arguments for specific modes

        Returns:
            torch.Tensor: Initialized tensor
        """
        torch_device_str = torch_device_map[device]
        torch_dtype = to_torch_dtype(dtype)

        # Handle non-contiguous tensors
        if strides is not None:
            return TensorInitializer._create_strided_tensor(
                shape, strides, torch_dtype, torch_device_str, mode, **kwargs
            )
        else:
            return TensorInitializer._create_contiguous_tensor(
                shape, torch_dtype, torch_device_str, mode, **kwargs
            )

    @staticmethod
    def _create_contiguous_tensor(shape, torch_dtype, torch_device_str, mode, **kwargs):
        """Create contiguous tensor"""
        if is_integer_dtype(torch_dtype):
            return TensorInitializer._create_integer_tensor(
                shape, torch_dtype, torch_device_str, mode, **kwargs
            )
        elif is_complex_dtype(torch_dtype):
            return TensorInitializer._create_complex_tensor(
                shape, torch_dtype, torch_device_str, mode, **kwargs
            )

        # Handle real floating-point types
        if mode == TensorInitializer.RANDOM:
            return torch.rand(shape, dtype=torch_dtype, device=torch_device_str)
        elif mode == TensorInitializer.ZEROS:
            return torch.zeros(shape, dtype=torch_dtype, device=torch_device_str)
        elif mode == TensorInitializer.ONES:
            return torch.ones(shape, dtype=torch_dtype, device=torch_device_str)
        elif mode == TensorInitializer.RANDINT:
            low = kwargs.get("low", -2000000000)
            high = kwargs.get("high", 2000000000)
            return torch.randint(
                low, high, shape, dtype=torch_dtype, device=torch_device_str
            )
        elif mode == TensorInitializer.MANUAL:
            tensor = kwargs.get("set_tensor")
            if tensor is None:
                raise ValueError("Manual mode requires set_tensor")
            if list(tensor.shape) != list(shape):
                raise ValueError(
                    f"Shape mismatch: expected {shape}, got {tensor.shape}"
                )
            return tensor.to(torch_dtype).to(torch_device_str)
        elif mode == TensorInitializer.BINARY:
            tensor = kwargs.get("set_tensor")
            if tensor is None:
                raise ValueError("Binary mode requires set_tensor")
            return tensor.to(torch_dtype).to(torch_device_str)
        elif mode == TensorInitializer.FROM_FILE:
            return TensorInitializer._load_from_file(
                kwargs.get("file_path"), shape, torch_dtype, torch_device_str
            )
        else:
            raise ValueError(f"Unsupported initialization mode: {mode}")

    @staticmethod
    def _create_integer_tensor(shape, torch_dtype, torch_device_str, mode, **kwargs):
        """Create integer tensor"""
        if mode == TensorInitializer.RANDOM:
            if torch_dtype == torch.bool:
                return torch.randint(
                    0, 2, shape, dtype=torch_dtype, device=torch_device_str
                ).bool()
            elif torch_dtype == torch.uint8:
                return torch.randint(
                    0, 256, shape, dtype=torch_dtype, device=torch_device_str
                )
            else:
                dtype_info = torch.iinfo(torch_dtype)
                return torch.randint(
                    dtype_info.min,
                    dtype_info.max,
                    shape,
                    dtype=torch_dtype,
                    device=torch_device_str,
                )
        elif mode == TensorInitializer.ZEROS:
            return torch.zeros(shape, dtype=torch_dtype, device=torch_device_str)
        elif mode == TensorInitializer.ONES:
            return torch.ones(shape, dtype=torch_dtype, device=torch_device_str)
        elif mode == TensorInitializer.RANDINT:
            low = kwargs.get("low", -100)
            high = kwargs.get("high", 100)
            return torch.randint(
                low, high, shape, dtype=torch_dtype, device=torch_device_str
            )
        elif mode == TensorInitializer.MANUAL:
            tensor = kwargs.get("set_tensor")
            if tensor is None:
                raise ValueError("Manual mode requires set_tensor")
            if list(tensor.shape) != list(shape):
                raise ValueError(
                    f"Shape mismatch: expected {shape}, got {tensor.shape}"
                )
            return tensor.to(torch_dtype).to(torch_device_str)
        elif mode == TensorInitializer.BINARY:
            tensor = kwargs.get("set_tensor")
            if tensor is None:
                raise ValueError("Binary mode requires set_tensor")
            return tensor.to(torch_dtype).to(torch_device_str)
        else:
            return torch.randint(
                0, 100, shape, dtype=torch_dtype, device=torch_device_str
            )

    @staticmethod
    def _create_complex_tensor(shape, torch_dtype, torch_device_str, mode, **kwargs):
        """Create complex tensor (complex64 or complex128)"""
        if mode == TensorInitializer.RANDOM:
            # Create complex tensor with random real and imaginary parts
            real_part = torch.rand(shape, device=torch_device_str)
            imag_part = torch.rand(shape, device=torch_device_str)
            complex_tensor = torch.complex(real_part, imag_part)
            return complex_tensor.to(torch_dtype)
        elif mode == TensorInitializer.ZEROS:
            return torch.zeros(shape, dtype=torch_dtype, device=torch_device_str)
        elif mode == TensorInitializer.ONES:
            return torch.ones(shape, dtype=torch_dtype, device=torch_device_str)
        elif mode == TensorInitializer.MANUAL:
            tensor = kwargs.get("set_tensor")
            if tensor is None:
                raise ValueError("Manual mode requires set_tensor")
            if list(tensor.shape) != list(shape):
                raise ValueError(
                    f"Shape mismatch: expected {shape}, got {tensor.shape}"
                )
            return tensor.to(torch_dtype).to(torch_device_str)
        elif mode == TensorInitializer.BINARY:
            tensor = kwargs.get("set_tensor")
            if tensor is None:
                raise ValueError("Binary mode requires set_tensor")
            return tensor.to(torch_dtype).to(torch_device_str)
        else:
            # Default to random complex values
            real_part = torch.rand(shape, device=torch_device_str)
            imag_part = torch.rand(shape, device=torch_device_str)
            complex_tensor = torch.complex(real_part, imag_part)
            return complex_tensor.to(torch_dtype)

    @staticmethod
    def _create_strided_tensor(
        shape, strides, torch_dtype, torch_device_str, mode, **kwargs
    ):
        """Create non-contiguous tensor with specific strides"""
        # Calculate required storage size
        storage_size = 0
        for i in range(len(shape)):
            if shape[i] > 0:
                storage_size += (shape[i] - 1) * abs(strides[i])
        storage_size += 1

        # Create base storage
        base_tensor = TensorInitializer._create_contiguous_tensor(
            (storage_size,), torch_dtype, torch_device_str, mode, **kwargs
        )

        # Create strided view
        return torch.as_strided(base_tensor, shape, strides)

    @staticmethod
    def _load_from_file(file_path, shape_or_size, torch_dtype, torch_device_str):
        """Load tensor data from file"""
        if file_path is None:
            raise ValueError("FROM_FILE mode requires file_path")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = file_path.suffix.lower()

        if file_extension in [".pt", ".pth"]:
            tensor = torch.load(file_path, map_location=torch_device_str)
        elif file_extension in [".bin", ".dat", ".raw"]:
            tensor = TensorInitializer._load_binary_file(
                file_path, shape_or_size, torch_dtype, torch_device_str
            )
        elif file_extension in [".npy"]:
            try:
                import numpy as np

                numpy_array = np.load(file_path)
                tensor = (
                    torch.from_numpy(numpy_array).to(torch_dtype).to(torch_device_str)
                )
            except ImportError:
                raise ImportError("NumPy is required to load .npy files")
        else:
            try:
                tensor = torch.load(file_path, map_location=torch_device_str)
            except:
                tensor = TensorInitializer._load_binary_file(
                    file_path, shape_or_size, torch_dtype, torch_device_str
                )

        tensor = tensor.to(torch_dtype).to(torch_device_str)

        # Validate shape/size
        if isinstance(shape_or_size, (list, tuple)):
            if list(tensor.shape) != list(shape_or_size):
                raise ValueError(
                    f"Shape mismatch: expected {shape_or_size}, got {tensor.shape}"
                )
        else:
            if tensor.numel() != shape_or_size:
                raise ValueError(
                    f"Size mismatch: expected {shape_or_size} elements, got {tensor.numel()}"
                )

        return tensor

    @staticmethod
    def _load_binary_file(file_path, shape_or_size, torch_dtype, torch_device_str):
        """Load tensor from raw binary file"""
        with open(file_path, "rb") as f:
            binary_data = f.read()

        if isinstance(shape_or_size, (list, tuple)):
            tensor = torch.frombuffer(binary_data, dtype=torch_dtype).reshape(
                shape_or_size
            )
        else:
            tensor = torch.frombuffer(binary_data, dtype=torch_dtype)

        return tensor.to(torch_device_str)


class TensorSpec:
    """Unified tensor specification for both contiguous and non-contiguous tensors"""

    def __init__(
        self,
        shape=None,
        dtype=None,
        strides=None,
        value=None,
        is_scalar=False,
        init_mode=TensorInitializer.RANDOM,
        **kwargs,
    ):
        self.shape = shape
        self.dtype = dtype
        self.strides = strides
        self.value = value
        self.is_scalar = is_scalar
        self.init_mode = init_mode
        self.kwargs = kwargs

    @classmethod
    def from_tensor(
        cls,
        shape,
        strides=None,
        dtype=None,
        init_mode=TensorInitializer.RANDOM,
        **kwargs,
    ):
        """Create tensor specification - unified interface for both contiguous and non-contiguous"""
        return cls(
            shape=shape,
            dtype=dtype,
            strides=strides,
            is_scalar=False,
            init_mode=init_mode,
            **kwargs,
        )

    @classmethod
    def from_scalar(cls, value, dtype=None):
        return cls(value=value, dtype=dtype, is_scalar=True)

    @classmethod
    def from_strided_tensor(
        cls,
        shape,
        strides,
        dtype=None,
        init_mode=TensorInitializer.RANDOM,
        **kwargs,
    ):
        """Alias for from_tensor with explicit strides (for backward compatibility)"""
        return cls.from_tensor(shape, strides, dtype, init_mode, **kwargs)

    def create_torch_tensor(self, device):
        """Create a torch tensor based on this specification"""
        if self.is_scalar:
            return self.value

        # Create tensor using unified interface
        return TensorInitializer.create_tensor(
            shape=self.shape,
            dtype=self.dtype,
            device=device,
            mode=self.init_mode,
            strides=self.strides,
            **self.kwargs,
        )

    def is_tensor_input(self):
        """Check if this spec represents a tensor input (not scalar)"""
        return not self.is_scalar

    def __str__(self):
        if self.is_scalar:
            return f"scalar({self.value})"
        else:
            strides_str = f", strides={self.strides}" if self.strides else ""
            dtype_str = f", dtype={self.dtype}" if self.dtype else ""
            return f"tensor{self.shape}{strides_str}{dtype_str}"
