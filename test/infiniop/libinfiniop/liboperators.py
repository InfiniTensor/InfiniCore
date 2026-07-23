import ctypes
import os
import platform
from ctypes import POINTER, c_int, c_int64, c_uint64
from pathlib import Path

from .datatypes import *
from .devices import *
from .op_register import OpRegister
from .structs import *

INFINI_ROOT = os.getenv("INFINI_ROOT") or str(Path.home() / ".infini")
INFINI_RT_ROOT = os.getenv("INFINI_RT_ROOT")


class InfiniLib:
    def __init__(self, librt, libop):
        self.librt = librt
        self.libop = libop

    def __getattr__(self, name):
        if hasattr(self.libop, name):
            return getattr(self.libop, name)
        elif hasattr(self.librt, name):
            return getattr(self.librt, name)
        else:
            raise AttributeError(f"Attribute {name} not found in library")


# Open operators library
def open_lib():
    def find_library(root, subdirs, library_name):
        for subdir in subdirs:
            full_path = os.path.join(root, subdir, library_name)
            if os.path.isfile(full_path):
                return full_path
        return None

    system_name = platform.system()
    # Load the library
    if system_name == "Windows":
        libop_path = find_library(INFINI_ROOT, ("bin",), "infiniop.dll")
        librt_root = INFINI_RT_ROOT or INFINI_ROOT
        librt_path = find_library(librt_root, ("bin",), "infinirt.dll")
    elif system_name == "Linux":
        libop_path = find_library(INFINI_ROOT, ("lib", "lib64"), "libinfiniop.so")
        librt_root = INFINI_RT_ROOT or INFINI_ROOT
        librt_path = find_library(
            librt_root, ("lib", "lib64"), "libinfinirt.so"
        )
    else:
        raise RuntimeError(f"Unsupported platform: {system_name}")

    assert libop_path is not None, (
        "Cannot find infiniop.dll or libinfiniop.so. Check if INFINI_ROOT is set correctly."
    )
    assert librt_path is not None, (
        "Cannot find infinirt.dll or libinfinirt.so. Check INFINI_RT_ROOT or INFINI_ROOT."
    )

    librt = ctypes.CDLL(librt_path)
    libop = ctypes.CDLL(libop_path)
    lib = InfiniLib(librt, libop)
    lib.infiniopCreateTensorDescriptor.argtypes = [
        POINTER(infiniopTensorDescriptor_t),
        c_uint64,
        POINTER(c_uint64),
        POINTER(c_int64),
        c_int,
    ]
    lib.infiniopCreateTensorDescriptor.restype = c_int
    lib.infiniopDestroyTensorDescriptor.argtypes = [infiniopTensorDescriptor_t]
    lib.infiniopDestroyTensorDescriptor.restype = c_int
    lib.infiniopSetRuntimeDevice.argtypes = [c_int, c_int]
    lib.infiniopSetRuntimeDevice.restype = c_int
    lib.infiniopCreateHandle.argtypes = [POINTER(infiniopHandle_t)]
    lib.infiniopCreateHandle.restype = c_int
    lib.infiniopDestroyHandle.argtypes = [infiniopHandle_t]
    lib.infiniopDestroyHandle.restype = c_int
    OpRegister.register_lib(lib)

    return lib


LIBINFINIOP = open_lib()
