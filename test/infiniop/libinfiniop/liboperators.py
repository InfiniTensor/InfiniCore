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
    def find_libraries_for_platform():
        """查找当前平台的库文件并返回路径"""
        system_name = platform.system()

        # 确定库文件的目录和名称
        if system_name == "Windows":
            lib_dir = "bin"
            op_lib_name = "infiniop.dll"
            rt_lib_name = "infinirt.dll"
        elif system_name == "Linux":
            lib_dir = "lib"
            op_lib_name = "libinfiniop.so"
            rt_lib_name = "libinfinirt.so"
        elif system_name == "Darwin":
            lib_dir = "lib"
            op_lib_name = "libinfiniop.dylib"
            rt_lib_name = "libinfinirt.dylib"
        else:
            raise RuntimeError(f"Unsupported operating system: {system_name}")

        # 构建完整路径
        op_path = os.path.join(INFINI_ROOT, lib_dir, op_lib_name)
        rt_path = os.path.join(INFINI_ROOT, lib_dir, rt_lib_name)

        # 检查文件是否存在
        if not os.path.isfile(op_path):
            raise RuntimeError(f"Cannot find the InfiniOP library ({op_lib_name}). Searched in {os.path.join(INFINI_ROOT, lib_dir)}. Check if INFINI_ROOT is set correctly.")
        if not os.path.isfile(rt_path):
            raise RuntimeError(f"Cannot find the InfiniRT library ({rt_lib_name}). Searched in {os.path.join(INFINI_ROOT, lib_dir)}. Check if INFINI_ROOT is set correctly.")

        print(f"[Info] Found libraries in {INFINI_ROOT}/{lib_dir}: {op_lib_name}, {rt_lib_name}")
        return op_path, rt_path

    # 查找库文件
    libop_path, librt_path = find_libraries_for_platform()

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
