import torch
from pathlib import Path
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
    FROM_FILE = "from_file"

    @staticmethod
    def create_tensor(
        shape, dtype, device, mode=RANDOM, strides=None, set_tensor=None, file_path=None
    ):
        """
        Create a torch tensor with specified initialization mode

        Args:
            shape: Tensor shape
            dtype: infinicore dtype
            device: InfiniDeviceEnum
            mode: Initialization mode
            strides: Optional strides for strided tensors
            set_tensor: Pre-existing tensor for manual/binary mode
            file_path: Path to file for FROM_FILE mode

        Returns:
            torch.Tensor: Initialized tensor
        """
        # Convert InfiniDeviceEnum to torch device string
        torch_device_str = torch_device_map[device]
        torch_dtype = to_torch_dtype(dtype)

        # Handle strided tensors - calculate required storage size
        if strides is not None:
            # Calculate the required storage size for strided tensor
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
            elif mode == TensorInitializer.FROM_FILE:
                base_tensor = TensorInitializer._load_from_file(
                    file_path, storage_size, torch_dtype, torch_device_str
                )
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
            elif mode == TensorInitializer.FROM_FILE:
                tensor = TensorInitializer._load_from_file(
                    file_path, shape, torch_dtype, torch_device_str
                )
            else:
                raise ValueError(f"Unsupported initialization mode: {mode}")

        return tensor

    @staticmethod
    def _load_from_file(file_path, shape_or_size, torch_dtype, torch_device_str):
        """
        Load tensor data from file using PyTorch's native methods

        Args:
            file_path: Path to the file
            shape_or_size: Tensor shape for contiguous or size for strided
            torch_dtype: Target torch dtype
            torch_device_str: Target device string

        Returns:
            torch.Tensor: Tensor with data loaded from file
        """
        if file_path is None:
            raise ValueError("FROM_FILE mode requires file_path")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type and load accordingly
        file_extension = file_path.suffix.lower()

        if file_extension in [".pt", ".pth"]:
            # PyTorch native format
            tensor = torch.load(file_path, map_location=torch_device_str)

        elif file_extension in [".bin", ".dat", ".raw"]:
            # Raw binary format - we need to know the expected shape
            tensor = TensorInitializer._load_binary_file(
                file_path, shape_or_size, torch_dtype, torch_device_str
            )

        elif file_extension in [".npy"]:
            # NumPy format - fallback to numpy if needed
            try:
                import numpy as np

                numpy_array = np.load(file_path)
                tensor = (
                    torch.from_numpy(numpy_array).to(torch_dtype).to(torch_device_str)
                )
            except ImportError:
                raise ImportError("NumPy is required to load .npy files")

        else:
            # Try to load as PyTorch format first, then fallback to binary
            try:
                tensor = torch.load(file_path, map_location=torch_device_str)
            except:
                # Fallback to binary loading
                tensor = TensorInitializer._load_binary_file(
                    file_path, shape_or_size, torch_dtype, torch_device_str
                )

        # Ensure correct dtype and device
        tensor = tensor.to(torch_dtype).to(torch_device_str)

        # Validate shape/size
        if isinstance(shape_or_size, (list, tuple)):
            # Contiguous tensor - check shape
            if list(tensor.shape) != list(shape_or_size):
                raise ValueError(
                    f"Tensor shape mismatch: expected {shape_or_size}, got {tensor.shape}"
                )
        else:
            # Strided tensor - check total size
            if tensor.numel() != shape_or_size:
                raise ValueError(
                    f"Tensor size mismatch: expected {shape_or_size} elements, got {tensor.numel()}"
                )

        return tensor

    @staticmethod
    def _load_binary_file(file_path, shape_or_size, torch_dtype, torch_device_str):
        """
        Load tensor from raw binary file

        Args:
            file_path: Path to binary file
            shape_or_size: Expected shape or size
            torch_dtype: Target dtype
            torch_device_str: Target device

        Returns:
            torch.Tensor: Loaded tensor
        """
        # Read binary data
        with open(file_path, "rb") as f:
            binary_data = f.read()

        # Create tensor from buffer
        if isinstance(shape_or_size, (list, tuple)):
            # Contiguous tensor with known shape
            tensor = torch.frombuffer(binary_data, dtype=torch_dtype).reshape(
                shape_or_size
            )
        else:
            # Strided tensor - just 1D buffer
            tensor = torch.frombuffer(binary_data, dtype=torch_dtype)

        return tensor.to(torch_device_str)

    @staticmethod
    def save_to_file(tensor, file_path, format="auto"):
        """
        Save tensor data to file using PyTorch's native methods

        Args:
            tensor: torch.Tensor to save
            file_path: Path to save the file
            format: File format ('auto', 'torch', 'binary', 'numpy')
        """
        file_path = Path(file_path)

        if format == "auto":
            # Determine format from file extension
            file_extension = file_path.suffix.lower()
            if file_extension in [".pt", ".pth"]:
                format = "torch"
            elif file_extension in [".npy"]:
                format = "numpy"
            else:
                format = "binary"

        if format == "torch":
            # PyTorch native format (preserves metadata)
            torch.save(tensor, file_path)

        elif format == "binary":
            # Raw binary format
            with open(file_path, "wb") as f:
                f.write(tensor.cpu().numpy().tobytes())

        elif format == "numpy":
            # NumPy format
            try:
                import numpy as np

                np.save(file_path, tensor.cpu().numpy())
            except ImportError:
                raise ImportError("NumPy is required to save .npy files")

        else:
            raise ValueError(f"Unsupported format: {format}")

        print(
            f"Tensor saved to {file_path} (shape: {tensor.shape}, dtype: {tensor.dtype}, format: {format})"
        )

    @staticmethod
    def list_supported_formats():
        """Return list of supported file formats"""
        return {
            "torch": [".pt", ".pth"],  # PyTorch native format
            "binary": [".bin", ".dat", ".raw"],  # Raw binary
            "numpy": [".npy"],  # NumPy format
        }


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
        file_path=None,  # For FROM_FILE mode
        file_format=None,  # Optional file format hint
    ):
        self.shape = shape
        self.dtype = dtype
        self.strides = strides
        self.value = value
        self.is_scalar = is_scalar
        self.is_contiguous = is_contiguous
        self.init_mode = init_mode
        self.custom_tensor = custom_tensor
        self.file_path = file_path
        self.file_format = file_format

    @classmethod
    def from_tensor(
        cls,
        shape,
        dtype=None,
        strides=None,
        is_contiguous=True,
        init_mode=TensorInitializer.RANDOM,
        custom_tensor=None,
        file_path=None,
    ):
        return cls(
            shape=shape,
            dtype=dtype,
            strides=strides,
            is_scalar=False,
            is_contiguous=is_contiguous,
            init_mode=init_mode,
            custom_tensor=custom_tensor,
            file_path=file_path,
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
        file_path=None,
    ):
        return cls(
            shape=shape,
            dtype=dtype,
            strides=strides,
            is_scalar=False,
            is_contiguous=False,
            init_mode=init_mode,
            custom_tensor=custom_tensor,
            file_path=file_path,
        )

    @classmethod
    def from_file(
        cls,
        file_path,
        shape,
        dtype=None,
        strides=None,
        is_contiguous=True,
        file_format=None,
    ):
        """
        Create TensorSpec that loads data from file

        Args:
            file_path: Path to file
            shape: Tensor shape
            dtype: infinicore dtype (inferred from file if None)
            strides: Optional strides for strided tensors
            is_contiguous: Whether tensor is contiguous
            file_format: Optional file format hint

        Returns:
            TensorSpec: Configured for file loading
        """
        return cls(
            shape=shape,
            dtype=dtype,
            strides=strides,
            is_scalar=False,
            is_contiguous=is_contiguous,
            init_mode=TensorInitializer.FROM_FILE,
            file_path=file_path,
            file_format=file_format,
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
            file_path=self.file_path,
        )
