import infinicore.device
from infinicore.lib import _infinicore


def get_device():
    """Get the current active device.

    Returns:
        device: The current active device object
    """
    return _infinicore.get_device()


def get_device_count(device_type):
    """Get the number of available devices of a specific type.

    Args:
        device_type (str): The type of device to count (e.g., "cuda", "cpu", "npu")

    Returns:
        int: The number of available devices of the specified type
    """
    return _infinicore.get_device_count(infinicore.device(device_type)._underlying.type)


def set_device(device):
    """Set the current active device.

    Args:
        device: The device to set as active
    """
    _infinicore.set_device(device._underlying)


def sync_stream():
    """Synchronize the current stream."""
    _infinicore.sync_stream()


def sync_device():
    """Synchronize the current device."""
    _infinicore.sync_device()


def get_stream():
    """Get the current stream.

    Returns:
        stream: The current stream object
    """
    return _infinicore.get_stream()
