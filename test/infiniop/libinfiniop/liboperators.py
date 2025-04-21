import os
import platform
import ctypes
from ctypes import c_int, c_int64, c_uint64, Structure, POINTER
from .datatypes import *
from .devices import *
from pathlib import Path

Device = c_int
Optype = c_int

INFINI_ROOT = os.getenv("INFINI_ROOT") or str(Path.home() / ".infini")


class TensorDescriptor(Structure):
    _fields_ = []


infiniopTensorDescriptor_t = ctypes.POINTER(TensorDescriptor)


class CTensor:
    def __init__(self, desc, torch_tensor):
        self.descriptor = desc
        self.torch_tensor_ = torch_tensor
        self.data = torch_tensor.data_ptr()

    def destroyDesc(self, lib_):
        lib_.infiniopDestroyTensorDescriptor(self.descriptor)
        self.descriptor = None


class Handle(Structure):
    _fields_ = [("device", c_int), ("device_id", c_int)]


infiniopHandle_t = POINTER(Handle)


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


def open_lib():
  
    libop_path = None
    librt_path = None

    def find_library_in_infini_root(subdir, library_name):
        full_path = os.path.join(INFINI_ROOT, subdir, library_name)
        print(f"[Debug] Checking for library at: {full_path}") 
        if os.path.isfile(full_path):
            return full_path
        return None
    # --- End Helper ---

    system_name = platform.system()
    print(f"[Debug] Detected system: {system_name}")
    print(f"[Debug] Using INFINI_ROOT: {INFINI_ROOT}") 


    if system_name == "Windows":
        libop_path = find_library_in_infini_root("bin", "infiniop.dll")
        librt_path = find_library_in_infini_root("bin", "infinirt.dll")
    elif system_name == "Linux":
        libop_path = find_library_in_infini_root("lib", "libinfiniop.so")
        librt_path = find_library_in_infini_root("lib", "libinfinirt.so")
    elif system_name == "Darwin":  # <--- Added macOS ("Darwin") case
        libop_path = find_library_in_infini_root("lib", "libinfiniop.dylib")
        librt_path = find_library_in_infini_root("lib", "libinfinirt.dylib")
    else:
        raise RuntimeError(f"Unsupported operating system: {system_name}")

    assert (
        libop_path is not None
    ), f"Cannot find the InfiniOP library ({'infiniop.dll' if system_name == 'Windows' else 'libinfiniop.so' if system_name == 'Linux' else 'libinfiniop.dylib'}). Searched in {os.path.join(INFINI_ROOT, 'bin' if system_name == 'Windows' else 'lib')}. Check if INFINI_ROOT is set correctly and the library exists."
    assert (
        librt_path is not None
    ), f"Cannot find the InfiniRT library ({'infinirt.dll' if system_name == 'Windows' else 'libinfinirt.so' if system_name == 'Linux' else 'libinfinirt.dylib'}). Searched in {os.path.join(INFINI_ROOT, 'bin' if system_name == 'Windows' else 'lib')}. Check if INFINI_ROOT is set correctly and the library exists."

    print(f"[Info] Loading InfiniRT from: {librt_path}")
    print(f"[Info] Loading InfiniOP from: {libop_path}") 

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
    lib.infiniopCreateHandle.argtypes = [POINTER(infiniopHandle_t)]
    lib.infiniopCreateHandle.restype = c_int
    lib.infiniopDestroyHandle.argtypes = [infiniopHandle_t]
    lib.infiniopDestroyHandle.restype = c_int
    lib.infinirtSetDevice.argtypes = [c_int, c_int]
    lib.infinirtSetDevice.restype = c_int

    return lib