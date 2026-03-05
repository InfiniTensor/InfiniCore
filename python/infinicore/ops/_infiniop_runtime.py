import ctypes
import os
import platform
from ctypes import POINTER, Structure, c_double, c_int, c_int64, c_size_t, c_ssize_t
from pathlib import Path


class TensorDescriptor(Structure):
    _fields_ = []


infiniopTensorDescriptor_t = POINTER(TensorDescriptor)


class Handle(Structure):
    _fields_ = [("device", c_int), ("device_id", c_int)]


infiniopHandle_t = POINTER(Handle)


class OpDescriptor(Structure):
    _fields_ = [("device", c_int), ("device_id", c_int)]


infiniopOperatorDescriptor_t = POINTER(OpDescriptor)


class _InfiniLib:
    def __init__(self, librt, libop):
        self._librt = librt
        self._libop = libop

    def __getattr__(self, name):
        if hasattr(self._libop, name):
            return getattr(self._libop, name)
        if hasattr(self._librt, name):
            return getattr(self._librt, name)
        raise AttributeError(f"{name} not found in Infini libraries")


_LIB = None
_HANDLE_CACHE = {}  # (device_type:int, device_id:int) -> infiniopHandle_t


def _check_error(status: int) -> None:
    if status != 0:
        raise RuntimeError(f"Infini runtime/operator call failed with status={status}")


def _infini_root() -> str:
    return os.getenv("INFINI_ROOT") or str(Path.home() / ".infini")


def _load_lib() -> _InfiniLib:
    global _LIB
    if _LIB is not None:
        return _LIB

    infini_root = _infini_root()

    system_name = platform.system()
    if system_name == "Windows":
        libop_path = os.path.join(infini_root, "bin", "infiniop.dll")
        librt_path = os.path.join(infini_root, "bin", "infinirt.dll")
    elif system_name == "Linux":
        libop_path = os.path.join(infini_root, "lib", "libinfiniop.so")
        librt_path = os.path.join(infini_root, "lib", "libinfinirt.so")
    else:
        raise RuntimeError(f"Unsupported platform: {system_name}")

    if not os.path.isfile(libop_path):
        raise FileNotFoundError(
            f"Cannot find InfiniOP library at {libop_path} (check INFINI_ROOT)"
        )
    if not os.path.isfile(librt_path):
        raise FileNotFoundError(
            f"Cannot find InfiniRT library at {librt_path} (check INFINI_ROOT)"
        )

    librt = ctypes.CDLL(librt_path)
    libop = ctypes.CDLL(libop_path)
    lib = _InfiniLib(librt, libop)

    # Core runtime and descriptor APIs
    lib.infiniopCreateTensorDescriptor.argtypes = [
        POINTER(infiniopTensorDescriptor_t),
        c_size_t,
        POINTER(c_size_t),
        POINTER(c_ssize_t),
        c_int,
    ]
    lib.infiniopCreateTensorDescriptor.restype = c_int
    lib.infiniopDestroyTensorDescriptor.argtypes = [infiniopTensorDescriptor_t]
    lib.infiniopDestroyTensorDescriptor.restype = c_int

    lib.infiniopCreateHandle.argtypes = [POINTER(infiniopHandle_t)]
    lib.infiniopCreateHandle.restype = c_int
    lib.infiniopDestroyHandle.argtypes = [infiniopHandle_t]
    lib.infiniopDestroyHandle.restype = c_int

    lib.infinirtSetDevice.argtypes = [c_int, c_int]
    lib.infinirtSetDevice.restype = c_int

    # Operator APIs used by the benchmark runner.
    # log10
    lib.infiniopCreateLog10Descriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopCreateLog10Descriptor.restype = c_int
    lib.infiniopGetLog10WorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopGetLog10WorkspaceSize.restype = c_int
    lib.infiniopLog10.argtypes = [
        infiniopOperatorDescriptor_t,
        ctypes.c_void_p,
        c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.infiniopLog10.restype = c_int
    lib.infiniopDestroyLog10Descriptor.argtypes = [infiniopOperatorDescriptor_t]
    lib.infiniopDestroyLog10Descriptor.restype = c_int

    # log1p
    lib.infiniopCreateLog1pDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopCreateLog1pDescriptor.restype = c_int
    lib.infiniopGetLog1pWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopGetLog1pWorkspaceSize.restype = c_int
    lib.infiniopLog1p.argtypes = [
        infiniopOperatorDescriptor_t,
        ctypes.c_void_p,
        c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.infiniopLog1p.restype = c_int
    lib.infiniopDestroyLog1pDescriptor.argtypes = [infiniopOperatorDescriptor_t]
    lib.infiniopDestroyLog1pDescriptor.restype = c_int

    # histc
    lib.infiniopCreateHistcDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int64,
        c_double,
        c_double,
    ]
    lib.infiniopCreateHistcDescriptor.restype = c_int
    lib.infiniopGetHistcWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopGetHistcWorkspaceSize.restype = c_int
    lib.infiniopHistc.argtypes = [
        infiniopOperatorDescriptor_t,
        ctypes.c_void_p,
        c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.infiniopHistc.restype = c_int
    lib.infiniopDestroyHistcDescriptor.argtypes = [infiniopOperatorDescriptor_t]
    lib.infiniopDestroyHistcDescriptor.restype = c_int

    # dot
    lib.infiniopCreateDotDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopCreateDotDescriptor.restype = c_int
    lib.infiniopGetDotWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopGetDotWorkspaceSize.restype = c_int
    lib.infiniopDot.argtypes = [
        infiniopOperatorDescriptor_t,
        ctypes.c_void_p,
        c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.infiniopDot.restype = c_int
    lib.infiniopDestroyDotDescriptor.argtypes = [infiniopOperatorDescriptor_t]
    lib.infiniopDestroyDotDescriptor.restype = c_int

    # avg_pool3d
    lib.infiniopCreateAvgPool3dDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        ctypes.c_void_p,  # kernel_size
        ctypes.c_void_p,  # stride (nullable)
        ctypes.c_void_p,  # padding
    ]
    lib.infiniopCreateAvgPool3dDescriptor.restype = c_int
    lib.infiniopGetAvgPool3dWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopGetAvgPool3dWorkspaceSize.restype = c_int
    lib.infiniopAvgPool3d.argtypes = [
        infiniopOperatorDescriptor_t,
        ctypes.c_void_p,
        c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.infiniopAvgPool3d.restype = c_int
    lib.infiniopDestroyAvgPool3dDescriptor.argtypes = [infiniopOperatorDescriptor_t]
    lib.infiniopDestroyAvgPool3dDescriptor.restype = c_int

    _LIB = lib
    return _LIB


def _ensure_handle(device_type: int, device_id: int) -> infiniopHandle_t:
    lib = _load_lib()

    # Always set the device for the underlying runtime before invoking operators.
    _check_error(lib.infinirtSetDevice(c_int(device_type), c_int(device_id)))

    key = (int(device_type), int(device_id))
    handle = _HANDLE_CACHE.get(key)
    if handle is not None:
        return handle

    handle = infiniopHandle_t()
    _check_error(lib.infiniopCreateHandle(ctypes.byref(handle)))
    _HANDLE_CACHE[key] = handle
    return handle


def handle_for_tensor(tensor) -> infiniopHandle_t:
    # infinicore.device has an underlying C++ device instance.
    dev = tensor.device._underlying
    return _ensure_handle(int(dev.type), int(dev.index))


def create_tensor_descriptor(tensor) -> infiniopTensorDescriptor_t:
    lib = _load_lib()

    shape = list(tensor.shape)
    strides = list(tensor.stride())
    ndim = len(shape)

    c_shape = (c_size_t * ndim)(*shape)
    c_strides = (c_ssize_t * ndim)(*strides)

    desc = infiniopTensorDescriptor_t()
    _check_error(
        lib.infiniopCreateTensorDescriptor(
            ctypes.byref(desc),
            c_size_t(ndim),
            c_shape,
            c_strides,
            c_int(int(tensor.dtype._underlying)),
        )
    )
    return desc


def destroy_tensor_descriptor(desc: infiniopTensorDescriptor_t) -> None:
    lib = _load_lib()
    if desc:
        _check_error(lib.infiniopDestroyTensorDescriptor(desc))
