import safetensors.torch
import torch

def load_tensor_from_safetensors(file_path: str, tensor_name: str) -> torch.Tensor:
    """
    Loads a specific tensor by name from a safetensors file.

    Args:
        file_path (str): The path to the safetensors file.
        tensor_name (str): The name of the tensor to load.

    Returns:
        torch.Tensor: The loaded tensor.

    Raises:
        FileNotFoundError: If the safetensors file does not exist.
        KeyError: If the tensor_name is not found in the safetensors file.
    """
    try:
        with safetensors.torch.safe_open(file_path, framework="pt") as f:
            if tensor_name not in f.keys():
                raise KeyError(f"Tensor '{tensor_name}' not found in {file_path}")
            tensor = f.get_tensor(tensor_name)
            return tensor
    except Exception as e:
        raise RuntimeError(f"Failed to load tensor '{tensor_name}' from '{file_path}': {e}")
