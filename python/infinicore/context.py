from infinicore.lib import _infinicore
from infinicore.device import device


def setDevice(device_obj):
    """
    Set the current device context.

    Args:
        device_obj: A device object (e.g., from infinicore.device("cuda", 0))

    Example:
        >>> import infinicore
        >>> cuda_dev = infinicore.device("cuda", 0)
        >>> infinicore.context.setDevice(cuda_dev)
    """
    if isinstance(device_obj, device):
        _infinicore.set_device(device_obj._underlying)
    else:
        # Try to convert to device if it's not already
        dev = device(device_obj)
        _infinicore.set_device(dev._underlying)


def getDevice():
    """
    Get the current device context.

    Returns:
        A device object representing the current context device.

    Example:
        >>> import infinicore
        >>> current_dev = infinicore.context.getDevice()
        >>> print(current_dev)
        cuda:0
    """
    infinicore_device = _infinicore.get_device()
    return device._from_infinicore_device(infinicore_device)


def getDeviceCount(device_type):
    """
    Get the number of devices of a given type.

    Args:
        device_type: Device type (e.g., "cuda", "cpu", or Device.Type enum)

    Returns:
        Number of devices of the specified type.

    Example:
        >>> import infinicore
        >>> count = infinicore.context.getDeviceCount("cuda")
        >>> print(count)
        8
    """
    if isinstance(device_type, str):
        # Convert string to Device.Type
        type_map = {
            "cpu": _infinicore.Device.Type.CPU,
            "cuda": _infinicore.Device.Type.NVIDIA,
            "mlu": _infinicore.Device.Type.CAMBRICON,
            "npu": _infinicore.Device.Type.ASCEND,
            "musa": _infinicore.Device.Type.MOORE,
        }
        device_type = type_map.get(
            device_type.lower(), _infinicore.Device.Type.CPU)

    return _infinicore.get_device_count(device_type)
