"""
GGUF file reading support for InfiniCore.

This module provides Python bindings for reading GGUF format model files.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import gguf
except ImportError:
    raise ImportError(
        "gguf-py package is required. Install it with: pip install gguf"
    )


class GGUFReader:
    """Reader for GGUF format model files."""

    def __init__(self, filepath: str):
        """Initialize GGUF reader.

        Args:
            filepath: Path to the GGUF file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GGUF file not found: {filepath}")

        self.filepath = filepath
        self.reader = gguf.GGUFReader(filepath)
        self._tensor_info_map = {}
        self._metadata_map = {}

        # Build tensor info map
        for tensor in self.reader.tensors:
            self._tensor_info_map[tensor.name] = tensor

        # Build metadata map
        for key, field in self.reader.fields.items():
            self._metadata_map[key] = field

    def get_tensor_names(self) -> List[str]:
        """Get list of all tensor names in the file.

        Returns:
            List of tensor names.
        """
        return list(self._tensor_info_map.keys())

    def get_tensor_info(self, tensor_name: str) -> Dict:
        """Get information about a tensor.

        Args:
            tensor_name: Name of the tensor.

        Returns:
            Dictionary with tensor information (name, shape, dtype, etc.).
        """
        if tensor_name not in self._tensor_info_map:
            raise KeyError(f"Tensor not found: {tensor_name}")

        tensor = self._tensor_info_map[tensor_name]
        return {
            "name": tensor.name,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.tensor_type),
            "data_offset": tensor.data_offset,
            "n_elements": tensor.n_elements,
            "n_bytes": tensor.n_bytes,
        }

    def get_tensor_data(self, tensor_name: str) -> np.ndarray:
        """Get tensor data as numpy array.

        Args:
            tensor_name: Name of the tensor.

        Returns:
            Numpy array with tensor data.
        """
        if tensor_name not in self._tensor_info_map:
            raise KeyError(f"Tensor not found: {tensor_name}")

        tensor = self._tensor_info_map[tensor_name]
        # tensor.data is already a numpy array
        return tensor.data

    def get_metadata(self, key: str, default=None):
        """Get metadata value.

        Args:
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            Metadata value or default.
        """
        if key not in self._metadata_map:
            return default

        field = self._metadata_map[key]
        # Use contents() method to get the actual value
        try:
            return field.contents()
        except Exception:
            # Fallback to parts if contents() fails
            return field.parts[-1] if field.parts else default

    def is_split_file(self) -> bool:
        """Check if this is a split GGUF file.

        Returns:
            True if this is a split file, False otherwise.
        """
        return self.get_metadata("split.count") is not None

    def get_split_no(self) -> int:
        """Get split number (0-indexed).

        Returns:
            Split number, or 0 if not a split file.
        """
        return self.get_metadata("split.no", 0)

    def get_split_count(self) -> int:
        """Get total number of splits.

        Returns:
            Total number of splits, or 1 if not a split file.
        """
        return self.get_metadata("split.count", 1)

    def get_all_metadata(self) -> Dict:
        """Get all metadata as a dictionary.

        Returns:
            Dictionary of all metadata key-value pairs.
        """
        result = {}
        for key, field in self._metadata_map.items():
            try:
                result[key] = field.contents()
            except Exception:
                # Fallback to parts if contents() fails
                if field.parts:
                    result[key] = field.parts[-1]
        return result


def find_split_files(base_path: str) -> List[str]:
    """Find all split GGUF files for a given base path.

    Args:
        base_path: Base path (can be the first split file or base name).

    Returns:
        List of split file paths in order.
    """
    if not os.path.exists(base_path):
        # Try to find split files by pattern
        base_dir = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        if "." in base_name:
            base_name = ".".join(base_name.split(".")[:-1])  # Remove extension

        split_files = []
        i = 0
        while True:
            # Try different naming patterns
            patterns = [
                f"{base_name}.split.{i:05d}.gguf",
                f"{base_name}.{i:05d}-of-*.gguf",
                f"{base_name}.split{i}.gguf",
            ]

            found = False
            for pattern in patterns:
                import glob
                matches = glob.glob(os.path.join(base_dir, pattern))
                if matches:
                    split_files.append(matches[0])
                    found = True
                    break

            if not found:
                break
            i += 1

        if split_files:
            return sorted(split_files)

    # Try to read the first file to get split count
    try:
        reader = GGUFReader(base_path)
        if reader.is_split_file():
            split_count = reader.get_split_count()
            base_dir = os.path.dirname(base_path)
            base_name = os.path.basename(base_path)

            # Remove split suffix if present
            if ".split." in base_name:
                base_name = base_name.split(".split.")[0]
            elif ".00000-of-" in base_name:
                base_name = base_name.split(".00000-of-")[0]

            split_files = []
            for i in range(split_count):
                # Try different naming patterns
                patterns = [
                    f"{base_name}.split.{i:05d}.gguf",
                    f"{base_name}.{i:05d}-of-{split_count:05d}.gguf",
                    f"{base_name}.split{i}.gguf",
                ]

                for pattern in patterns:
                    filepath = os.path.join(base_dir, pattern)
                    if os.path.exists(filepath):
                        split_files.append(filepath)
                        break

            if len(split_files) == split_count:
                return sorted(split_files)
    except Exception:
        pass

    # If no splits found, return single file
    return [base_path] if os.path.exists(base_path) else []


def load_gguf_tensors(
    filepath: Union[str, List[str]],
    tensor_names: Optional[List[str]] = None,
    device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """Load tensors from GGUF file(s).

    Args:
        filepath: Path to GGUF file or list of split file paths.
        tensor_names: Optional list of tensor names to load. If None, loads all.
        device: Target device (currently only "cpu" supported).

    Returns:
        Dictionary mapping tensor names to numpy arrays.
    """
    if isinstance(filepath, str):
        filepaths = find_split_files(filepath)
    else:
        filepaths = filepath

    result = {}

    for filepath in filepaths:
        reader = GGUFReader(filepath)

        if tensor_names is None:
            tensor_names_to_load = reader.get_tensor_names()
        else:
            tensor_names_to_load = [name for name in tensor_names if name not in result]

        for tensor_name in tensor_names_to_load:
            if tensor_name in reader.get_tensor_names():
                result[tensor_name] = reader.get_tensor_data(tensor_name)

    return result
