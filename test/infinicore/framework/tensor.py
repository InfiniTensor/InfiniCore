import torch
from .datatypes import to_torch_dtype
from .devices import torch_device_map


class TensorInitializer:
    """Tensor data initializer with multiple modes"""

    RANDOM = "random"
    ZEROS = "zeros"
    ONES = "ones"
    RANDINT = "randint"
    MANUAL = "manual"
    BINARY = "binary"

    @staticmethod
    def create_tensor(shape, dtype, device, mode=RANDOM, strides=None, set_tensor=None):
        """
        Create a torch tensor with specified initialization mode

        Args:
            shape: Tensor shape
            dtype: infinicore dtype
            device: InfiniDeviceEnum
            mode: Initialization mode
            strides: Optional strides for strided tensors
            set_tensor: Pre-existing tensor for manual/binary mode

        Returns:
            torch.Tensor: Initialized tensor
        """
        # Convert InfiniDeviceEnum to torch device string
        torch_device_str = torch_device_map[device]
        torch_dtype = to_torch_dtype(dtype)

        # Handle strided tensors - calculate required storage size
        if strides is not None:
            # Calculate the required storage size for strided tensor
            # The storage size needed is: max(offset + 1) for all elements
            # where offset = sum(index[i] * stride[i] for i in range(len(shape)))
            # The maximum offset occurs at the last element: sum((shape[i]-1) * strides[i])
            storage_size = 0
            for i in range(len(shape)):
                if shape[i] > 0:
                    storage_size += (shape[i] - 1) * abs(strides[i])
            storage_size += 1  # Add 1 for the base element

            # Create base storage with sufficient size
            if mode == TensorInitializer.RANDOM:
                base_tensor = torch.rand(
                    storage_size, dtype=torch_dtype, device=torch_device_str
                )
            elif mode == TensorInitializer.ZEROS:
                base_tensor = torch.zeros(
                    storage_size, dtype=torch_dtype, device=torch_device_str
                )
            elif mode == TensorInitializer.ONES:
                base_tensor = torch.ones(
                    storage_size, dtype=torch_dtype, device=torch_device_str
                )
            elif mode == TensorInitializer.RANDINT:
                base_tensor = torch.randint(
                    -2000000000,
                    2000000000,
                    (storage_size,),
                    dtype=torch_dtype,
                    device=torch_device_str,
                )
            elif mode == TensorInitializer.MANUAL:
                assert set_tensor is not None, "Manual mode requires set_tensor"
                base_tensor = set_tensor.to(torch_dtype).to(torch_device_str)
            elif mode == TensorInitializer.BINARY:
                assert set_tensor is not None, "Binary mode requires set_tensor"
                base_tensor = set_tensor.to(torch_dtype).to(torch_device_str)
            else:
                raise ValueError(f"Unsupported initialization mode: {mode}")

            # Create strided view
            tensor = torch.as_strided(base_tensor, shape, strides)
        else:
            # Contiguous tensor
            if mode == TensorInitializer.RANDOM:
                tensor = torch.rand(shape, dtype=torch_dtype, device=torch_device_str)
            elif mode == TensorInitializer.ZEROS:
                tensor = torch.zeros(shape, dtype=torch_dtype, device=torch_device_str)
            elif mode == TensorInitializer.ONES:
                tensor = torch.ones(shape, dtype=torch_dtype, device=torch_device_str)
            elif mode == TensorInitializer.RANDINT:
                tensor = torch.randint(
                    -2000000000,
                    2000000000,
                    shape,
                    dtype=torch_dtype,
                    device=torch_device_str,
                )
            elif mode == TensorInitializer.MANUAL:
                assert set_tensor is not None, "Manual mode requires set_tensor"
                assert shape == list(set_tensor.shape), "Shape mismatch in manual mode"
                tensor = set_tensor.to(torch_dtype).to(torch_device_str)
            elif mode == TensorInitializer.BINARY:
                assert set_tensor is not None, "Binary mode requires set_tensor"
                assert shape == list(set_tensor.shape), "Shape mismatch in binary mode"
                tensor = set_tensor.to(torch_dtype).to(torch_device_str)
            else:
                raise ValueError(f"Unsupported initialization mode: {mode}")

        return tensor


class TensorSpec:
    """Tensor specification supporting various input types and per-tensor dtype"""

    def __init__(
        self,
        shape=None,
        dtype=None,
        strides=None,
        value=None,
        is_scalar=False,
        is_contiguous=True,
        init_mode=TensorInitializer.RANDOM,  # Default to random initialization
        custom_tensor=None,  # For manual/binary mode
    ):
        self.shape = shape
        self.dtype = dtype
        self.strides = strides
        self.value = value
        self.is_scalar = is_scalar
        self.is_contiguous = is_contiguous
        self.init_mode = init_mode
        self.custom_tensor = custom_tensor

    @classmethod
    def from_tensor(
        cls,
        shape,
        dtype=None,
        strides=None,
        is_contiguous=True,
        init_mode=TensorInitializer.RANDOM,
        custom_tensor=None,
    ):
        return cls(
            shape=shape,
            dtype=dtype,
            strides=strides,
            is_scalar=False,
            is_contiguous=is_contiguous,
            init_mode=init_mode,
            custom_tensor=custom_tensor,
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
        custom_tensor=None,
    ):
        return cls(
            shape=shape,
            dtype=dtype,
            strides=strides,
            is_scalar=False,
            is_contiguous=False,
            init_mode=init_mode,
            custom_tensor=custom_tensor,
        )

    def create_torch_tensor(self, device, dtype_config, tensor_index=0):
        """Create a torch tensor based on this specification"""
        if self.is_scalar:
            return self.value

        # Determine dtype - ensure we're using infinicore dtype, not torch dtype
        if self.dtype is not None:
            tensor_dtype = self.dtype
        elif isinstance(dtype_config, dict) and f"input_{tensor_index}" in dtype_config:
            tensor_dtype = dtype_config[f"input_{tensor_index}"]
        elif isinstance(dtype_config, (list, tuple)) and tensor_index < len(
            dtype_config
        ):
            tensor_dtype = dtype_config[tensor_index]
        else:
            tensor_dtype = dtype_config

        # Create tensor using the specified initialization mode
        return TensorInitializer.create_tensor(
            shape=self.shape,
            dtype=tensor_dtype,
            device=device,
            mode=self.init_mode,
            strides=self.strides,
            set_tensor=self.custom_tensor,
        )
